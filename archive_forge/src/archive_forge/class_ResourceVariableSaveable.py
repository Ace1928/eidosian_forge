import functools
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
class ResourceVariableSaveable(saveable_object.SaveableObject):
    """SaveableObject implementation that handles ResourceVariables."""

    def __init__(self, var, slice_spec, name):
        self._var_device = var.device
        self._var_shape = var.shape
        if isinstance(var, tensor_lib.Tensor):
            self.handle_op = var.op.inputs[0]
            tensor = var
        elif resource_variable_ops.is_resource_variable(var):

            def _read_variable_closure(v):

                def f():
                    with ops.device(v.device):
                        if context.executing_eagerly() and (not v.is_initialized()):
                            return None
                        x = v.read_value_no_copy()
                        with ops.device('/device:CPU:0'):
                            return array_ops.identity(x)
                return f
            self.handle_op = var.handle
            tensor = _read_variable_closure(var)
        else:
            raise ValueError(f'Saveable is neither a resource variable nor a read operation. Got: {repr(var)}')
        spec = saveable_object.SaveSpec(tensor, slice_spec, name, dtype=var.dtype, device=var.device)
        super(ResourceVariableSaveable, self).__init__(var, [spec], name)

    def restore(self, restored_tensors, restored_shapes):
        """Restores tensors. Raises ValueError if incompatible shape found."""
        restored_tensor = restored_tensors[0]
        if restored_shapes is not None:
            restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0])
        with ops.device(self._var_device):
            restored_tensor = array_ops.identity(restored_tensor)
            try:
                assigned_variable = resource_variable_ops.shape_safe_assign_variable_handle(self.handle_op, self._var_shape, restored_tensor)
            except ValueError as e:
                raise ValueError(f'Received incompatible tensor with shape {restored_tensor.shape} when attempting to restore variable with shape {self._var_shape} and name {self.name}.') from e
            return assigned_variable