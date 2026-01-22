import collections
import weakref
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.trackable import constants
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.tracking.CheckpointInitialValue', v1=[])
class CheckpointInitialValue(object):
    """Tensor wrapper for managing update UIDs in `Variables`.

  When supplied as an initial value, objects of this type let a `Variable`
  (`Variable`, `ResourceVariable`, etc.) know the UID of the restore the initial
  value came from. This allows deferred restorations to be sequenced in the
  order the user specified them, and lets us fall back on assignment if an
  initial value is not set (e.g. due to a custom getter interfering).

  See comments in _add_variable_with_custom_getter for more information about
  how `CheckpointInitialValue` is used.
  """

    def __init__(self, checkpoint_position, shape=None, shard_info=None):
        if shard_info:
            full_shape_str = ' '.join(('%d' % d for d in shape)) + ' '
            slice_spec = ':'.join(('%d,%d' % (o, s) for o, s in zip(shard_info.offset, shard_info.shape)))
            shape_and_slice = full_shape_str + slice_spec
        else:
            shape_and_slice = ''
        self.wrapped_value = checkpoint_position.value_tensors({VARIABLE_VALUE_KEY: shape_and_slice})[VARIABLE_VALUE_KEY]
        self._checkpoint_position = checkpoint_position

    def __tf_tensor__(self, dtype=None, name=None):
        del dtype
        del name
        return self.wrapped_value

    @property
    def checkpoint_position(self):
        return self._checkpoint_position