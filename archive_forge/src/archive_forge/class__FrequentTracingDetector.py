import dataclasses
import functools
import os
import threading
import types as types_lib
import weakref
from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.distribute.parallel_device import parallel_device
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import autograph_util
from tensorflow.python.eager.polymorphic_function import compiler_ir
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.tf_export import tf_export
class _FrequentTracingDetector(object):
    """Class keeping track of how many recent calls triggered tracing."""
    __slots__ = ['_calls_per_tracings', '_call_count', '_total_warning_count']

    def __init__(self):
        self._calls_per_tracings = []
        self._total_warning_count = 0
        self._call_count = 0

    def called_with_tracing(self, function_name, omit_warning):
        """Updates the list of most recent calls' tracing information.

    Warns the user when recent calls caused retracing too often.

    Args:
      function_name: the python function being traced.
      omit_warning: If 'True', this call will not warn the user even if
        retracing happens too often.
    """
        self._call_count += 1
        self._calls_per_tracings.append(1)
        while self._calls_per_tracings:
            if self._call_count - self._calls_per_tracings[0] > FREQUENT_TRACING_WARNING_MAX_CALL_HISTORY:
                self._call_count -= self._calls_per_tracings.pop(0)
            else:
                break
        if omit_warning or self._total_warning_count >= FREQUENT_TRACING_WARNING_MAX_WARNING_PER_DETECTOR:
            return
        if len(self._calls_per_tracings) >= FREQUENT_TRACING_WARNING_THRESHOLD:
            self._total_warning_count += 1
            logging.warning('{} out of the last {} calls to {} triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.'.format(len(self._calls_per_tracings), self._call_count, function_name))

    def called_without_tracing(self):
        if not self._calls_per_tracings:
            self._calls_per_tracings = [0]
        self._calls_per_tracings[-1] += 1
        self._call_count += 1