import abc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.saving.utils_v1 import signature_def_utils as unexported_signature_utils
from tensorflow.python.saved_model import signature_def_utils
class PredictOutput(ExportOutput):
    """Represents the output of a generic prediction head.

  A generic prediction need not be either a classification or a regression.

  Named outputs must be provided as a dict from string to `Tensor`,
  """
    _SINGLE_OUTPUT_DEFAULT_NAME = 'output'

    def __init__(self, outputs):
        """Constructor for PredictOutput.

    Args:
      outputs: A `Tensor` or a dict of string to `Tensor` representing the
        predictions.

    Raises:
      ValueError: if the outputs is not dict, or any of its keys are not
          strings, or any of its values are not `Tensor`s.
    """
        self._outputs = self._wrap_and_check_outputs(outputs, self._SINGLE_OUTPUT_DEFAULT_NAME, error_label='Prediction')

    @property
    def outputs(self):
        return self._outputs

    def as_signature_def(self, receiver_tensors):
        return signature_def_utils.predict_signature_def(receiver_tensors, self.outputs)