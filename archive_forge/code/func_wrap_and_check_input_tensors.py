from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_FEATURE_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_LABEL_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_RECEIVER_DEFAULT_NAME
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def wrap_and_check_input_tensors(tensors, field_name, allow_int_keys=False):
    """Ensure that tensors is a dict of str to Tensor mappings.

  Args:
    tensors: dict of `str` (or `int`s if `allow_int_keys=True`) to `Tensors`, or
      a single `Tensor`.
    field_name: name of the member field of `ServingInputReceiver` whose value
      is being passed to `tensors`.
    allow_int_keys: If set to true, the `tensor` dict keys may also be `int`s.

  Returns:
    dict of str to Tensors; this is the original dict if one was passed, or
    the original tensor wrapped in a dictionary.

  Raises:
    ValueError: if tensors is None, or has non-string keys,
      or non-Tensor values
  """
    if tensors is None:
        raise ValueError('{}s must be defined.'.format(field_name))
    if not isinstance(tensors, dict):
        tensors = {_SINGLE_TENSOR_DEFAULT_NAMES[field_name]: tensors}
    for name, tensor in tensors.items():
        _check_tensor_key(name, error_label=field_name, allow_ints=allow_int_keys)
        _check_tensor(tensor, name, error_label=field_name)
    return tensors