import abc
import math
import typing
from typing import Any, Dict, Callable, Iterable, List, Optional, Text, Tuple, TypeVar, Union
from absl import logging
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import ops
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
def _set_optimization_parameters(self, parameters: optimization_parameters_pb2.OptimizationParameters):
    parameters.simulated_quantization.enabled = True
    parameters.simulated_quantization.num_buckets = self.num_buckets
    parameters.simulated_quantization.clipping_limits.lower.value = self.lower
    parameters.simulated_quantization.clipping_limits.upper.value = self.upper