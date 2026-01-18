import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.module import module
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
def validate_synchronization_aggregation_trainable(synchronization, aggregation, trainable, name):
    """Given user-provided variable properties, sets defaults and validates."""
    if aggregation is None:
        aggregation = variables.VariableAggregation.NONE
    elif not isinstance(aggregation, (variables.VariableAggregation, variables.VariableAggregationV2)):
        try:
            aggregation = variables.VariableAggregationV2(aggregation)
        except ValueError:
            raise ValueError('Invalid variable aggregation mode: {} for variable: {}'.format(aggregation, name))
    if synchronization is None:
        synchronization = variables.VariableSynchronization.AUTO
    else:
        try:
            synchronization = variables.VariableSynchronization(synchronization)
        except ValueError:
            raise ValueError('Invalid variable synchronization mode: {} for variable: {}'.format(synchronization, name))
    if trainable is None:
        trainable = synchronization != variables.VariableSynchronization.ON_READ
    return (synchronization, aggregation, trainable)