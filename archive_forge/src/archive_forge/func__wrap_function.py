import weakref
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _wrap_function(self, fn, args=None, kwargs=None, signature=None, name=None):
    """Internal wrap function method with extended func_graph arguments."""
    fn_with_filter_and_scope, returned_ops = _filter_returned_ops(self._variable_holder.call_with_variable_creator_scope(fn))
    func_graph.func_graph_from_py_func(None, fn_with_filter_and_scope, args=args, kwargs=kwargs, signature=signature, add_control_dependencies=False, func_graph=self.graph)
    fn_inputs = self.graph.inputs[:-len(self.graph.captures)]
    flat_fn_outputs = nest.flatten(self.graph.structured_outputs)
    for index, op in returned_ops.items():
        flat_fn_outputs[index] = op
    fn_outputs = nest.pack_sequence_as(self.graph.structured_outputs, flat_fn_outputs)
    name = name or fn.__name__
    wrapped_function = self._wrapped_function.prune(fn_inputs, fn_outputs, name, self.graph.structured_input_signature)
    self._functions[name] = wrapped_function
    return wrapped_function