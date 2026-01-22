from tensorflow.python.framework import ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export
@classmethod
def _variable_call(cls, initial_value=None, trainable=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, import_scope=None, constraint=None, synchronization=variables.VariableSynchronization.AUTO, aggregation=variables.VariableAggregation.NONE, shape=None, experimental_enable_variable_lifting=None, expected_shape=None, collections=None, use_resource=None, **kwargs):
    """VariableV1 class getter. Useful to force the signature."""
    if cls is not VariableV1:
        return None
    previous_getter = lambda **kwargs: default_variable_creator(None, **kwargs)
    for _, getter in ops.get_default_graph()._variable_creator_stack:
        previous_getter = variables._make_getter(getter, previous_getter)
    if aggregation is None:
        aggregation = variables.VariableAggregation.NONE
    return previous_getter(initial_value=initial_value, trainable=trainable, validate_shape=validate_shape, caching_device=caching_device, name=name, variable_def=variable_def, dtype=dtype, import_scope=import_scope, constraint=constraint, synchronization=synchronization, aggregation=aggregation, shape=shape, experimental_enable_variable_lifting=experimental_enable_variable_lifting, expected_shape=expected_shape, collections=collections, use_resource=use_resource)