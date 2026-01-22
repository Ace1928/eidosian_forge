import contextlib
import dataclasses
import enum
import threading
from typing import Any, Callable, Dict, Optional, Tuple
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache as function_cache_lib
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import concrete_function as concrete_function_lib
from tensorflow.python.eager.polymorphic_function import function_context
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.util import compat
def _set_arg_keywords(concrete_function):
    """Sets arg keywords for ConcreteFunction."""
    seen_names = set()
    concrete_function._arg_keywords = []
    prefix_counts = {}
    graph = concrete_function.graph
    num_captures = len(graph.internal_captures + graph.deferred_internal_captures)
    num_positional = len(graph.inputs) - num_captures
    for arg in concrete_function.graph.inputs[:num_positional]:
        try:
            user_arg_name = compat.as_str(arg.op.get_attr('_user_specified_name'))
        except ValueError:
            user_arg_name = 'tensor_arg'
        proposal = user_arg_name
        while proposal in seen_names:
            index = prefix_counts.get(user_arg_name, 1)
            proposal = '{}_{}'.format(user_arg_name, index)
            prefix_counts[user_arg_name] = index + 1
        seen_names.add(proposal)
        concrete_function._arg_keywords.append(proposal)
    concrete_function._num_positional_args = num_positional