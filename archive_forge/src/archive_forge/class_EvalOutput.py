import abc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.saving.utils_v1 import signature_def_utils as unexported_signature_utils
from tensorflow.python.saved_model import signature_def_utils
class EvalOutput(_SupervisedOutput):
    """Represents the output of a supervised eval process.

  This class generates the appropriate signature def for exporting
  eval output by type-checking and wrapping loss, predictions, and metrics
  values.
  """

    def _get_signature_def_fn(self):
        return unexported_signature_utils.supervised_eval_signature_def