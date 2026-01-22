from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
class TPUVariableMixin(object):
    """Mixin for TPU variables."""

    def __init__(self, *args, **kwargs):
        super(TPUVariableMixin, self).__init__(*args, **kwargs)
        if ops.executing_eagerly_outside_functions():
            self._handle_id = self._common_name + '_' + str(id(self._primary))
        else:
            self._handle_id = self._common_name

    def __getattr__(self, name):
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self).__getattr__(name)
        else:
            raise AttributeError(f'`TPUVariableMixin.{name}` not accessible within a TPU context.')

    def get(self):
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self).get()
        else:
            raise NotImplementedError('`TPUVariableMixin.get()` is not supported within a TPU context.')

    def _get_as_operand(self):
        return self.read_value()

    @property
    def handle(self):
        """The handle by which this variable can be accessed."""
        tpu_context = tpu_util.enclosing_tpu_context()
        if tpu_context is None or context.executing_eagerly():
            var = self._get_on_device_or_primary()
            if isinstance(var, packed.PackedVarAndDevice):
                return var.on_device_handle()
            else:
                return var.handle
        else:
            is_packed = self._packed_var is not None
            val = self._values
            if is_packed:
                val = [self._packed_var]
            return tpu_context.get_replicated_var_handle(self._common_name, self._handle_id, val, self._is_mirrored(), is_packed)

    @property
    def device(self):
        return self.handle.device

    def _read_variable_op(self):
        """Reads the value of this variable."""
        if self.trainable:
            tape.variable_accessed(self)
        handle = self.handle
        if getattr(handle, 'is_packed', False):
            with ops.device(self._get_on_device_or_primary().device):
                return gen_resource_variable_ops.read_variable_op(handle, self.dtype)
        else:
            return gen_resource_variable_ops.read_variable_op(handle, self.dtype)

    def read_value(self):
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self).read_value()
        else:
            return self._read_variable_op()

    def value(self):
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self).value()
        else:
            return self._read_variable_op()

    def _as_graph_element(self):
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self)._as_graph_element()
        else:
            return None

    @property
    def op(self):
        if values_util.is_saving_non_distributed():
            return self._primary.op
        return values.DistributedVarOp(self._primary.op.name, self._primary.op.graph, self._primary.op.traceback, self._primary.op.type)

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        """Converts a variable to a tensor."""
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self)._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)
        elif dtype is not None and dtype != self.dtype:
            return math_ops.cast(self.read_value(), dtype)
        else:
            return self.handle if as_ref else self.read_value()