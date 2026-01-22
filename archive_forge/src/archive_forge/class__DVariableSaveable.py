import functools
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util.tf_export import tf_export
class _DVariableSaveable(saveable_object.SaveableObject):
    """Class for defining how to save/restore DTensor variable."""

    def __init__(self, dvariable, name):
        with ops.device(dvariable.device):
            original_layout = api.fetch_layout(dvariable)
        self._original_layout = original_layout
        self._dvariable = dvariable

        def pack(tensors, layout):
            with ops.device(dvariable.device):
                return api.pack(tensors, layout)
        host_layout = layout_lib.Layout(original_layout.sharding_specs, original_layout.mesh.host_mesh())

        def get_host_dtensor():
            if original_layout.mesh.device_type().upper() != 'CPU':
                if context.executing_eagerly():
                    host_dtensor = api.pack(api.unpack(dvariable.read_value()), host_layout)
                else:
                    host_dtensor = api.copy_to_mesh(dvariable.read_value(), host_layout)
            else:
                host_dtensor = dvariable.read_value()
            return math_ops.cast(host_dtensor, dtypes.bfloat16) if self.should_cast(host_dtensor) else host_dtensor
        num_local_devices = original_layout.mesh.num_local_devices()
        super(_DVariableSaveable, self).__init__(None, [DSaveSpec(tensor=get_host_dtensor, slice_spec=pack([''] * num_local_devices, layout_lib.Layout.replicated(original_layout.mesh.host_mesh(), rank=0)), name=pack([name] * num_local_devices, layout_lib.Layout.replicated(original_layout.mesh.host_mesh(), rank=0)), global_shape=dvariable.shape, layout=host_layout.to_string(), dtype=dtypes.bfloat16 if self.should_cast(dvariable) else dvariable.dtype, device=dvariable.device)], name)

    def should_cast(self, v):
        """Returns True if v has float32 dtype and is intructed to save as bf16.

    Args:
      v : The variable that determines whether to cast.

    Returns:
      True if current savable DVariable is instructed to save as bfloat16 and
        the variable has dtype float32.
    """
        return self._dvariable.save_as_bf16 and v.dtype == dtypes.float32

    def restore(self, restored_tensors, restored_shapes):
        """Restores the same value into all variables."""
        tensor, = restored_tensors

        @def_function.function
        def _restore(t):
            with ops.device(self._dvariable.device):
                return api.copy_to_mesh(t, self._original_layout)
        if self._original_layout.mesh.device_type().upper() != 'CPU':
            tensor = _restore(tensor)
        return self._dvariable.assign(math_ops.cast(tensor, dtype=self._dvariable.dtype) if self._dvariable.save_as_bf16 else tensor)