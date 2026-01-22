from typing import Generator, Optional, Text
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
def _get_custom_getter():
    """Returns a custom getter that this class's methods must be called under.

  All methods of this class must be called under a variable scope that was
  passed this custom getter. Example:

  ```python
  network = ConvNetBuilder(...)
  with tf.compat.v1.variable_scope('cg',
                                   custom_getter=network.get_custom_getter()):
    network.conv(...)
    # Call more methods of network here
  ```

  Currently, this custom getter only does anything if self.use_tf_layers is
  True. In that case, it causes variables to be stored as dtype
  self.variable_type, then casted to the requested dtype, instead of directly
  storing the variable as the requested dtype.
  """

    def inner_custom_getter(getter, *args, **kwargs):
        """Custom getter that forces variables to have type self.variable_type."""
        cast_to_bfloat16 = False
        requested_dtype = kwargs['dtype']
        if requested_dtype == dtypes.bfloat16:
            kwargs['dtype'] = dtypes.float32
            cast_to_bfloat16 = True
        var = getter(*args, **kwargs)
        if cast_to_bfloat16:
            var = math_ops.cast(var, dtypes.bfloat16)
        return var
    return inner_custom_getter