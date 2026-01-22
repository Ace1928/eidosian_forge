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
def _make_rnn_cell_fn(units, cell_type=_SIMPLE_RNN_KEY):
    """Convenience function to create `rnn_cell_fn` for canned RNN Estimators.

  Args:
    units: Iterable of integer number of hidden units per RNN layer.
    cell_type: A class producing a RNN cell or a string specifying the cell
      type. Supported strings are: `'simple_rnn'`, `'lstm'`, and `'gru'`.

  Returns:
    A function that returns a RNN cell.

  Raises:
    ValueError: If cell_type is not supported.
  """

    def rnn_cell_fn():
        cells = [_single_rnn_cell(n, cell_type) for n in units]
        if len(cells) == 1:
            return cells[0]
        return cells
    return rnn_cell_fn