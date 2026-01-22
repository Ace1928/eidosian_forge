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
class NoRestoreSaveable(saveable_object.SaveableObject):
    """Embeds a tensor in a checkpoint with no restore ops."""

    def __init__(self, tensor, name, dtype=None, device=None):
        spec = saveable_object.SaveSpec(tensor, '', name, dtype=dtype, device=device)
        super(NoRestoreSaveable, self).__init__(tensor, [spec], name)

    def restore(self, restored_tensors, restored_shapes):
        return gen_control_flow_ops.no_op()