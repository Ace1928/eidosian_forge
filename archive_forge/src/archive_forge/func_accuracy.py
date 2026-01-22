from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['metrics.accuracy'])
def accuracy(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """Calculates how often `predictions` matches `labels`.

  The `accuracy` function creates two local variables, `total` and
  `count` that are used to compute the frequency with which `predictions`
  matches `labels`. This frequency is ultimately returned as `accuracy`: an
  idempotent operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `accuracy`.
  Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
  where the corresponding elements of `predictions` and `labels` match and 0.0
  otherwise. Then `update_op` increments `total` with the reduced sum of the
  product of `weights` and `is_correct`, and it increments `count` with the
  reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose shape matches
      `predictions`.
    predictions: The predicted values, a `Tensor` of any shape.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `accuracy` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    accuracy: A `Tensor` representing the accuracy, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `accuracy`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.

  @compatibility(TF2)
  `tf.compat.v1.metrics.accuracy` is not compatible with eager
  execution or `tf.function`.
  Please use `tf.keras.metrics.Accuracy` instead for TF2 migration. After
  instantiating a `tf.keras.metrics.Accuracy` object, you can first call the
  `update_state()` method to record the prediction/labels, and then call the
  `result()` method to get the accuracy eagerly. You can also attach it to a
  Keras model when calling the `compile` method. Please refer to [this
  guide](https://www.tensorflow.org/guide/migrate#new-style_metrics_and_losses)
  for more details.

  #### Structural Mapping to Native TF2

  Before:

  ```python
  accuracy, update_op = tf.compat.v1.metrics.accuracy(
    labels=labels,
    predictions=predictions,
    weights=weights,
    metrics_collections=metrics_collections,
    update_collections=update_collections,
    name=name)
  ```

  After:

  ```python
   m = tf.keras.metrics.Accuracy(
     name=name,
     dtype=None)

   m.update_state(
   y_true=labels,
   y_pred=predictions,
   sample_weight=weights)

   accuracy = m.result()
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name    | Note                       |
  | :-------------------- | :-------------- | :------------------------- |
  | `label`               | `y_true`        | In `update_state()` method |
  | `predictions`         | `y_true`        | In `update_state()` method |
  | `weights`             | `sample_weight` | In `update_state()` method |
  | `metrics_collections` | Not supported   | Metrics should be tracked  |
  :                       :                 : explicitly or with Keras   :
  :                       :                 : APIs, for example,         :
  :                       :                 : [add_metric][add_metric],  :
  :                       :                 : instead of via collections :
  | `updates_collections` | Not supported   | -                          |
  | `name`                | `name`          | In constructor             |

  [add_metric]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_metric


  #### Before & After Usage Example

  Before:

  >>> g = tf.Graph()
  >>> with g.as_default():
  ...   logits = [1, 2, 3]
  ...   labels = [0, 2, 3]
  ...   acc, acc_op = tf.compat.v1.metrics.accuracy(logits, labels)
  ...   global_init = tf.compat.v1.global_variables_initializer()
  ...   local_init = tf.compat.v1.local_variables_initializer()
  >>> sess = tf.compat.v1.Session(graph=g)
  >>> sess.run([global_init, local_init])
  >>> print(sess.run([acc, acc_op]))
  [0.0, 0.66667]


  After:

  >>> m = tf.keras.metrics.Accuracy()
  >>> m.update_state([1, 2, 3], [0, 2, 3])
  >>> m.result().numpy()
  0.66667

  ```python
  # Used within Keras model
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.Accuracy()])
  ```

  @end_compatibility
  """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.accuracy is not supported when eager execution is enabled.')
    predictions, labels, weights = _remove_squeezable_dimensions(predictions=predictions, labels=labels, weights=weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
    is_correct = math_ops.cast(math_ops.equal(predictions, labels), dtypes.float32)
    return mean(is_correct, weights, metrics_collections, updates_collections, name or 'accuracy')