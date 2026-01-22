from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
from six import string_types
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.inputs.queues import feeding_functions
def _get_unique_target_key(features):
    """Returns a key not existed in the input dict `features`.

  Caller of `input_fn` usually provides `features` (dict of numpy arrays) and
  `target`, but the underlying feeding module expects a single dict of numpy
  arrays as input. So, the `target` needs to be packed into the `features`
  temporarily and unpacked after calling the feeding function. Toward this goal,
  this function returns a key not existed in the `features` to pack the
  `target`.

  Args:
    features: OrderedDict of numpy arrays

  Returns:
    A unique key that can be used to insert the subsequent target into
      features dict.
  """
    target_key = _TARGET_KEY
    while target_key in features:
        target_key += '_n'
    return target_key