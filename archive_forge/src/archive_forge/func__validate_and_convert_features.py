from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
from six import string_types
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.inputs.queues import feeding_functions
def _validate_and_convert_features(x):
    """Type check input data and make a shadow copy as an ordered dict.

  Args:
    x: numpy array object or dict of numpy array objects. If an array, the array
      will be treated as a single feature.

  Returns:
    OrderedDict copy of x.

  Raises:
    ValueError: if x is empty
    TypeError: if x is an unknown type.
  """
    if isinstance(x, dict):
        if not x:
            raise ValueError('x cannot be an empty dict')
        ordered_dict_data = collections.OrderedDict(sorted(x.items(), key=lambda t: t[0]))
    elif isinstance(x, np.ndarray):
        if x.size == 0:
            raise ValueError('x cannot be an empty array')
        ordered_dict_data = collections.OrderedDict({'__direct_np_input__': x})
    else:
        x_type = type(x).__name__
        raise TypeError('x must be a dict or array; got {}'.format(x_type))
    return ordered_dict_data