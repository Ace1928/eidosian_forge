from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import function as defun
from tensorflow.python.eager import wrap_function as wrap_function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
def wrap_cached_variables(concrete_function):
    """Wraps the concrete function if it uses cached read tensors.

  This function creates a new concrete function that captures variables
  instead of the cached read tensors.

  Args:
    concrete_function: A Concrete function that maybe captures cached read
      tensors.

  Returns:
    A concrete function that wraps the original concrete function, which
    captures variables instead. If the original function did not capture any
    cached values, then the function is not wrapped and the original object is
    returned.
  """
    outer_graph = func_graph_module.FuncGraph('{}_no_cache'.format(concrete_function.graph.name))
    mapped_captures = None
    remapped_captures = {}
    with outer_graph.as_default():
        for capture, placeholder in concrete_function.graph.captures:
            cached_variable = getattr(capture, '_cached_variable', None)
            if cached_variable is None:
                continue
            cached_variable = cached_variable()
            new_cached_value = cached_variable.read_value()
            key = id(capture)
            external = concrete_function.graph.function_captures.by_val_external[key]
            internal = concrete_function.graph.function_captures.by_val_internal[key]
            remapped_captures[key] = [external, internal]
            concrete_function.graph.function_captures.add_or_replace(key=key, external=new_cached_value, internal=placeholder, is_by_ref=False)
            mapped_captures = True
    if not mapped_captures:
        return concrete_function
    inner_concrete = defun.ConcreteFunction.from_func_graph(concrete_function.graph, concrete_function.function_type, {})

    def wrap_function(*args):
        return inner_concrete._call_flat(list(args), inner_concrete.captured_inputs)
    args = nest.flatten(concrete_function.structured_input_signature, expand_composites=True)
    func_graph_module.func_graph_from_py_func(None, wrap_function, args=tuple(args), kwargs={}, func_graph=outer_graph)
    fn = defun.ConcreteFunction.from_func_graph(outer_graph, concrete_function.function_type, {})
    fn._arg_keywords = concrete_function._arg_keywords
    fn._num_positional_args = concrete_function._num_positional_args
    for key, capture in remapped_captures.items():
        external, internal = capture
        concrete_function.graph._function_captures.add_or_replace(key=key, external=external, internal=internal, is_by_ref=False)
    return fn