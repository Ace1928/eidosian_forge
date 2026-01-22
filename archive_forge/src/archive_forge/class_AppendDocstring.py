import functools
import hashlib
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import tf_inspect
class AppendDocstring:
    """Helper class to promote private subclass docstring to public counterpart.

  Example:

  ```python
  class TransformedDistribution(Distribution):
    @distribution_util.AppendDocstring(
      additional_note="A special note!",
      kwargs_dict={"foo": "An extra arg."})
    def _prob(self, y, foo=None):
      pass
  ```

  In this case, the `AppendDocstring` decorator appends the `additional_note` to
  the docstring of `prob` (not `_prob`) and adds a new `kwargs`
  section with each dictionary item as a bullet-point.

  For a more detailed example, see `TransformedDistribution`.
  """

    def __init__(self, additional_note='', kwargs_dict=None):
        """Initializes the AppendDocstring object.

    Args:
      additional_note: Python string added as additional docstring to public
        version of function.
      kwargs_dict: Python string/string dictionary representing specific kwargs
        expanded from the **kwargs input.

    Raises:
      ValueError: if kwargs_dict.key contains whitespace.
      ValueError: if kwargs_dict.value contains newlines.
    """
        self._additional_note = additional_note
        if kwargs_dict:
            bullets = []
            for key in sorted(kwargs_dict.keys()):
                value = kwargs_dict[key]
                if any((x.isspace() for x in key)):
                    raise ValueError('Parameter name "%s" contains whitespace.' % key)
                value = value.lstrip()
                if '\n' in value:
                    raise ValueError('Parameter description for "%s" contains newlines.' % key)
                bullets.append('*  `%s`: %s' % (key, value))
            self._additional_note += '\n\n##### `kwargs`:\n\n' + '\n'.join(bullets)

    def __call__(self, fn):

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            return fn(*args, **kwargs)
        if _fn.__doc__ is None:
            _fn.__doc__ = self._additional_note
        else:
            _fn.__doc__ += '\n%s' % self._additional_note
        return _fn