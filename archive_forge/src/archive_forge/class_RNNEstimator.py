from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head as binary_head_lib
from tensorflow_estimator.python.estimator.head import multi_class_head as multi_head_lib
from tensorflow_estimator.python.estimator.head import sequential_head as seq_head_lib
@estimator_export('estimator.experimental.RNNEstimator', v1=[])
class RNNEstimator(estimator.Estimator):
    """An Estimator for TensorFlow RNN models with user-specified head.

  Example:

  ```python
  token_sequence = sequence_categorical_column_with_hash_bucket(...)
  token_emb = embedding_column(categorical_column=token_sequence, ...)

  estimator = RNNEstimator(
      head=tf.estimator.RegressionHead(),
      sequence_feature_columns=[token_emb],
      units=[32, 16], cell_type='lstm')

  # Or with custom RNN cell:
  def rnn_cell_fn(_):
    cells = [ tf.keras.layers.LSTMCell(size) for size in [32, 16] ]
    return tf.keras.layers.StackedRNNCells(cells)

  estimator = RNNEstimator(
      head=tf.estimator.RegressionHead(),
      sequence_feature_columns=[token_emb],
      rnn_cell_fn=rnn_cell_fn)

  # Input builders
  def input_fn_train: # returns x, y
    pass
  estimator.train(input_fn=input_fn_train, steps=100)

  def input_fn_eval: # returns x, y
    pass
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  def input_fn_predict: # returns x, None
    pass
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * if the head's `weight_column` is not `None`, a feature with
    `key=weight_column` whose value is a `Tensor`.
  * for each `column` in `sequence_feature_columns`:
    - a feature with `key=column.name` whose `value` is a `SparseTensor`.
  * for each `column` in `context_feature_columns`:
    - if `column` is a `CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss and predicted output are determined by the specified head.

  @compatibility(eager)
  Estimators are not compatible with eager execution.
  @end_compatibility
  """

    def __init__(self, head, sequence_feature_columns, context_feature_columns=None, units=None, cell_type=USE_DEFAULT, rnn_cell_fn=None, return_sequences=False, model_dir=None, optimizer='Adagrad', config=None):
        """Initializes a `RNNEstimator` instance.

    Args:
      head: A `Head` instance. This specifies the model's output and loss
        function to be optimized.
      sequence_feature_columns: An iterable containing the `FeatureColumn`s that
        represent sequential input. All items in the set should either be
        sequence columns (e.g. `sequence_numeric_column`) or constructed from
        one (e.g. `embedding_column` with `sequence_categorical_column_*` as
        input).
      context_feature_columns: An iterable containing the `FeatureColumn`s for
        contextual input. The data represented by these columns will be
        replicated and given to the RNN at each timestep. These columns must be
        instances of classes derived from `DenseColumn` such as
        `numeric_column`, not the sequential variants.
      units: Iterable of integer number of hidden units per RNN layer. If set,
        `cell_type` must also be specified and `rnn_cell_fn` must be `None`.
      cell_type: A class producing a RNN cell or a string specifying the cell
        type. Supported strings are: `'simple_rnn'`, `'lstm'`, and `'gru'`. If
          set, `units` must also be specified and `rnn_cell_fn` must be `None`.
      rnn_cell_fn: A function that returns a RNN cell instance that will be used
        to construct the RNN. If set, `units` and `cell_type` cannot be set.
        This is for advanced users who need additional customization beyond
        `units` and `cell_type`. Note that `tf.keras.layers.StackedRNNCells` is
        needed for stacked RNNs.
      return_sequences: A boolean indicating whether to return the last output
        in the output sequence, or the full sequence.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      optimizer: An instance of `tf.Optimizer` or string specifying optimizer
        type. Defaults to Adagrad optimizer.
      config: `RunConfig` object to configure the runtime settings.

    Note that a RNN cell has:
      - a `call` method.
      - a `state_size` attribute.
      - a `output_size` attribute.
      - a `get_initial_state` method.

    See the documentation on `tf.keras.layers.RNN` for more details.

    Raises:
      ValueError: If `units`, `cell_type`, and `rnn_cell_fn` are not
        compatible.
    """
        if return_sequences and (not isinstance(head, seq_head_lib._SequentialHead)):
            raise ValueError('Provided head must be a `_SequentialHead` object when `return_sequences` is set to True.')
        _verify_rnn_cell_input(rnn_cell_fn, units, cell_type)

        def _model_fn(features, labels, mode, config):
            """RNNEstimator model function."""
            del config
            rnn_layer = _make_rnn_layer(rnn_cell_fn=rnn_cell_fn, units=units, cell_type=cell_type, return_sequences=return_sequences)
            rnn_model = RNNModel(rnn_layer=rnn_layer, units=head.logits_dimension, sequence_feature_columns=sequence_feature_columns, context_feature_columns=context_feature_columns, return_sequences=return_sequences, name='rnn_model')
            return _get_rnn_estimator_spec(features, labels, mode, head=head, rnn_model=rnn_model, optimizer=optimizer, return_sequences=return_sequences)
        super(RNNEstimator, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config)