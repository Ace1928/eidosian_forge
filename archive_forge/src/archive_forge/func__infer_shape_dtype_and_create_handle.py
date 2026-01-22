import threading
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def _infer_shape_dtype_and_create_handle(initial_value, shape, dtype, name):
    """Infer shape and dtype from initial_value and create a variable handle."""
    with ops.name_scope(name, 'Variable', skip_on_eager=False) as name:
        handle_name = ops.name_from_scope_name(name)
        unique_id = '%s_%d' % (handle_name, ops.uid())
        device_context_manager = ops.NullContextmanager
        attr = attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(s=[compat.as_bytes(f'loc:@{handle_name}')]))
        with ops.get_default_graph()._attr_scope({'_class': attr}):
            with ops.name_scope('Initializer'), device_context_manager(None):
                if not callable(initial_value):
                    if isinstance(initial_value, trackable.CheckpointInitialValue):
                        raise NotImplementedError('CheckpointInitialValue is not supported to be the initial value of a lazy variable.')
                    initial_value = ops.convert_to_tensor(initial_value, name='initial_value', dtype=dtype)
                    assert not callable(initial_value)
                    assert initial_value.shape.is_compatible_with(shape)
                    dtype = dtype or initial_value.dtype.base_dtype
                    shape = shape or initial_value.shape
            assert dtype
            assert shape
            handle = resource_variable_ops._variable_handle_from_shape_and_dtype(shape=shape, dtype=dtype, shared_name=None, name=name, graph_mode=False, initial_value=None)
    return (initial_value, shape, dtype, handle, handle_name, unique_id)