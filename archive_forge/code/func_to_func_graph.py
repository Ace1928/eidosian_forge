import dataclasses
import traceback
import typing
from typing import Any, Dict, List, Optional, Sequence, Union
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_stack
def to_func_graph(atomic: AtomicFunction) -> func_graph_module.FuncGraph:
    """Generate a FuncGraph from an AtomicFunction."""
    input_signature, output_signature = function_type_lib.to_structured_signature(atomic.function_type)
    with ops.Graph().as_default():
        for f in atomic.children:
            ops.get_default_graph()._add_function(f)
        result = function_def_to_graph.function_def_to_graph(atomic.definition, structured_input_signature=input_signature, structured_outputs=output_signature, propagate_device_spec=True, include_library_functions=False)
        for f in atomic.children:
            result._add_function(f)
    for i, input_type in enumerate(atomic.function_type.flat_inputs):
        handle_data = input_type.dtype._handle_data
        if handle_data:
            handle_data_util.set_handle_data(result.inputs[i], handle_data.shape_inference)
        result.inputs[i].set_shape(input_type.shape)
    for i, output_type in enumerate(atomic.function_type.flat_outputs):
        handle_data = output_type.dtype._handle_data
        if handle_data:
            handle_data_util.set_handle_data(result.outputs[i], handle_data.shape_inference)
        result.outputs[i].set_shape(output_type.shape)
    result.collective_manager_ids_used = (atomic.call_options.collective_manager_ids_used,)
    return result