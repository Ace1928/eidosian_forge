import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
def _get_optimization_handler(optimization_parameters):
    """Gets the optimization handler given the parameter type."""
    if isinstance(optimization_parameters, AdagradParameters):
        return _AdagradHandler(optimization_parameters)
    elif isinstance(optimization_parameters, AdagradMomentumParameters):
        return _AdagradMomentumHandler(optimization_parameters)
    elif isinstance(optimization_parameters, ProximalAdagradParameters):
        return _ProximalAdagradHandler(optimization_parameters)
    elif isinstance(optimization_parameters, AdamParameters):
        return _AdamHandler(optimization_parameters)
    elif isinstance(optimization_parameters, FtrlParameters):
        return _FtrlHandler(optimization_parameters)
    elif isinstance(optimization_parameters, ProximalYogiParameters):
        return _ProximalYogiHandler(optimization_parameters)
    elif isinstance(optimization_parameters, StochasticGradientDescentParameters):
        return _StochasticGradientDescentHandler(optimization_parameters)
    elif isinstance(optimization_parameters, MomentumParameters):
        return _MomentumHandler(optimization_parameters)
    elif isinstance(optimization_parameters, RMSPropParameters):
        return _RMSPropHandler(optimization_parameters)
    elif isinstance(optimization_parameters, FrequencyEstimatorParameters):
        return _FrequencyEstimatorHandler(optimization_parameters)
    return NotImplementedError()