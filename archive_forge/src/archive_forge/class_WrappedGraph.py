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
class WrappedGraph(object):
    """Class for wrapping multiple TF 1.X functions in a single graph.

  Maintains a dictionary mapping names to wrapped functions. See
  `tf.compat.v1.wrap_function` to learn more about wrapping V1 functions.

  Functions wrapped using this class have access to variables and collections
  created in other wrapped functions, using the standard TF 1.X API (
  `tf.compat.v1.get_variable` or
  `tf.compat.v1.get_default_graph().get_collection(...)`)

  Outside a function, variables and collections may be accessed using the
  `variables` and `graph` properties.

  Example:

  ```
  def add_v1(x):
    with tf.compat.v1.variable_scope('vars', reuse=tf.compat.v1.AUTO_REUSE):
      v = tf.compat.v1.get_variable('v', shape=[], dtype=tf.int32)
    return v + x

  def increment_var_v1(x):
    with tf.compat.v1.variable_scope('vars', reuse=tf.compat.v1.AUTO_REUSE):
      v = tf.compat.v1.get_variable('v', shape=[], dtype=tf.int32)
    return v.assign_add(x)

  g = WrappedGraph()
  add = g.wrap_function(add_v1, [tf.TensorSpec([], tf.int32)])
  increment_var = g.wrap_function(increment_var_v1,
                                  [tf.TensorSpec([], tf.int32)])

  assert len(g.variables) == 1
  assert g.variables[0].numpy() == 0
  increment_var(tf.constant(5))
  assert g.variables[0].numpy() == 5

  ```
  """

    def __init__(self, variable_holder=None, **kwargs):
        self._variable_holder = variable_holder or VariableHolder(share_variables=True)
        name = kwargs.pop('name', 'wrapped_function_graph')
        collections = kwargs.pop('collections', {})
        self.graph = func_graph.FuncGraph(name, collections=collections, **kwargs)
        self._wrapped_function = WrappedFunction(self.graph, self._variable_holder)
        self._functions = {}

    @property
    def functions(self):
        return self._functions

    @property
    def variables(self):
        return self._variable_holder.variables

    def wrap_function(self, fn, signature, name=None):
        """Wraps a TF 1.X function and returns an eager-compatible function.

    All functions wrapped in the same `WrappedGraph` will have access to the
    same graph (`tf.compat.v1.get_default_graph` to get the graph object
    within a function, or `WrappedGraph.graph` to get the graph outside a
    function). Variables created within the function will be added to the
    `variables` list.

    Function inputs: All inputs to the function must be tensors (nested ok),
    with their shapes and dtypes defined in the `signature` argument.

    Function outputs:

      * The 1.X function may return tensors, variables, and ops. The wrapped
        eager-compatible function will always return tensors in the same nested
        structure.
      * Variables are replaced with a tensor containing the latest read values.
      * Returned ops are executed, and replaced with None.
      * The order of op execution and variable reads in the return is
        nondeterministic. For example:

        ```
        def update_var(x):
          v = tf.Variable(0)
          op = tf.compat.v1.assign(v, x).op
          return v, op

        g = WrappedGraph()
        fn = g.wrap_function(update_var)
        read_value, _ = fn(tf.constant(3))
        print(read_value.numpy())  # could be 0 or 3
        print(g.variables[0].numpy()) # always 3
        ```

    To ensure that ops in the function are executed (e.g. ops added to the
    `tf.GraphKeys.UPDATE_OPS` collection), include them in the function returns.

    Args:
      fn: a 1.X tensorflow function.
      signature: a possibly nested sequence of `TensorSpecs` specifying the
        shapes and dtypes of the arguments.
      name: an optional string name for the function. The function will be saved
        with key `name` in the `functions` dictionary.

    Returns:
      An eager-compatible function.
    """
        return self._wrap_function(fn, signature=signature, name=name)

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