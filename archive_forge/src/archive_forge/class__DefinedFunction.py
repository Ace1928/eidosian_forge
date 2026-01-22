import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
class _DefinedFunction(object):
    """_DefinedFunction encapsulates a function definition and its properties.

  Attributes:
    name: The function name.
    definition: The definition of this function. A FunctionDef proto.
    cached_definition: Same as definition. Needed to match AtomicFunction API.
    grad_func_name: If not None, the name of this function's gradient function.
    python_grad_func: A python callable implementing the gradient of
      the function python-side.
  """

    def __init__(self, func, argnames, input_types, func_name=None, grad_func=None, python_grad_func=None, out_names=None, shape_func=None, capture_by_value=False, allowlisted_stateful_ops=None, capture_resource_var_by_value=True, **kwargs):
        """Creates _DefinedFunction.

    Args:
      func:  A python callable which constructs a tf function body.
      argnames: A list of strings for function argument names.
      input_types: The function's argument types. Can be a tuple, list of
        tf data types.
      func_name: The function name. Defaults to None, in which derives from
        'func'.
      grad_func: This function's gradient function, if not None. Defaults
        to None.
      python_grad_func: A python callable implementing the gradient of
        the function python-side.
      out_names: An optional list of strings for the function return value
        names.
      shape_func: An optional function mapping an op to a list of static
        output shapes.
      capture_by_value: Boolean (defaults to False). If True, captured values
        will be copied into the function body.
      allowlisted_stateful_ops: A set of ops that if stateful we ignore and
        copy into the function body, when `capture_by_value` is True.
      capture_resource_var_by_value: Boolean (defaults to True). If False,
        captured resource variable returns the handle instead of value.
      **kwargs: The keyword arguments. **kwargs is passed to every call
        site of this function.

    Raises:
      ValueError: The function definition is invalid.

    """
        self._func = func
        self._input_types = input_types
        self._func_name = func_name
        self._grad_func = grad_func
        self._python_grad_func = python_grad_func
        self._out_names = out_names
        self._shape_func = shape_func
        self._capture_by_value = capture_by_value
        self._allowlisted_stateful_ops = allowlisted_stateful_ops
        if self._allowlisted_stateful_ops is None:
            self._allowlisted_stateful_ops = set()
        self._capture_resource_var_by_value = capture_resource_var_by_value
        self._extra_kwargs = kwargs
        self._definition = None
        self._c_func = None
        self._function_deleter = None
        self._sub_functions = {}
        device_funcs = ops.get_default_graph()._device_functions_outer_to_inner
        self._caller_device = device_funcs[-1] if device_funcs else None
        self._op_def = None
        assert isinstance(input_types, (list, tuple))
        self._arg_types = input_types
        self._arg_names = [argnames[i] if i < len(argnames) else 'arg%d' % i for i in range(len(input_types))]

    @property
    def name(self):
        """Function name."""
        self._create_definition_if_needed()
        return self._func_name

    @property
    def cached_definition(self):
        return self.definition

    @property
    def definition(self):
        """Function definition proto."""
        self._create_definition_if_needed()
        if self._c_func:
            with c_api_util.tf_buffer() as buf:
                with self._c_func.get() as func:
                    c_api.TF_FunctionToFunctionDef(func, buf)
                    fdef = function_pb2.FunctionDef()
                    proto_data = c_api.TF_GetBuffer(buf)
                    fdef.ParseFromString(compat.as_bytes(proto_data))
                    with ops.init_scope():
                        if context.executing_eagerly():
                            context.add_c_function(func)
                            self._function_deleter = _DefinedFunctionDeleter(fdef.signature.name)
            return fdef
        return self._definition

    @property
    def _signature(self):
        self._create_definition_if_needed()
        return self._op_def

    def set_grad_func(self, grad_func):
        """Specifies the gradient function of this function."""
        assert not self._grad_func
        assert isinstance(grad_func, _DefinedFunction)
        self._grad_func = grad_func

    @property
    def grad_func_name(self):
        """Returns the name of the gradient function."""
        return self._grad_func.name if self._grad_func else None

    @property
    def python_grad_func(self):
        """Python gradient function callable."""
        return self._python_grad_func

    @property
    def declared_input_types(self):
        """Returns the list of data types of explicit declared inputs."""
        return self._input_types

    @property
    def captured_inputs(self):
        """Returns the list of implicitly captured inputs."""
        self._create_definition_if_needed()
        return self._extra_inputs

    @property
    def stateful_ops(self):
        """Returns the list of stateful ops in function definition.

    Returns:
      A list of (op.name, op.type) pairs.
    """
        self._create_definition_if_needed()
        return self._stateful_ops

    def _create_definition_if_needed(self):
        """Creates the function definition if it's not created yet."""
        with context.graph_mode():
            self._create_definition_if_needed_impl()

    def _create_definition_if_needed_impl(self):
        """This is not what you want, see _create_definition_if_needed."""
        if self._definition is not None or self._c_func is not None:
            return
        variable_keys = []
        variable_keys.extend(ops.GraphKeys._VARIABLE_COLLECTIONS)
        variable_keys.append(vs._VARSTORE_KEY)
        parent_graph = ops.get_default_graph()
        collections_ref = {key: parent_graph.get_collection_ref(key) for key in variable_keys}
        temp_graph = func_graph_from_py_func(self._func, self._arg_names, self._arg_types, self._func_name, self._capture_by_value, self._caller_device, collections_ref=collections_ref, allowlisted_stateful_ops=self._allowlisted_stateful_ops, capture_resource_var_by_value=self._capture_resource_var_by_value)
        self._extra_inputs = temp_graph.extra_inputs
        self._sub_functions = temp_graph._functions
        if self._func_name:
            base_func_name = self._func_name
        else:
            base_func_name = function_utils.get_func_name(self._func)
            if self._grad_func:
                base_func_name += '_%s' % self._grad_func.name
        kwargs_attr = _parse_kwargs_as_attrs(base_func_name, **self._extra_kwargs)
        if not temp_graph._c_graph:
            self._definition = graph_to_function_def.graph_to_function_def(temp_graph, temp_graph.get_operations(), temp_graph.inputs, temp_graph.outputs, out_names=self._out_names)
            for k in kwargs_attr:
                self._definition.attr[k].CopyFrom(kwargs_attr[k])
            self._hash_str = self._create_hash_str(self._definition.signature.input_arg, self._definition.signature.output_arg, self._definition.node_def)
            if not self._func_name:
                self._func_name = '_'.join([base_func_name, self._hash_str])
            self._definition.signature.name = self._func_name
            if self._func.__doc__:
                self._definition.signature.description = self._func.__doc__
            self._op_def = self._definition.signature
        else:
            output_names = [compat.as_bytes(x) for x in self._out_names] if self._out_names else []
            description = self._func.__doc__ or None
            with temp_graph._c_graph.get() as c_graph:
                c_func = c_api.TF_GraphToFunction_wrapper(c_graph, base_func_name, self._func_name is None, None, [t._as_tf_output() for t in temp_graph.inputs], [t._as_tf_output() for t in temp_graph.outputs], output_names, [], [], None, description)
            self._c_func = c_api_util.ScopedTFFunction(c_func, base_func_name)
            self._set_c_attrs(kwargs_attr)
            self._op_def = self.definition.signature
            if self._func_name:
                assert self._func_name == self._op_def.name
            else:
                self._func_name = compat.as_str(self._op_def.name)
        self._stateful_ops = [(op.name, op.type) for op in temp_graph.get_operations() if op._is_stateful]

    def _set_c_attrs(self, attrs):
        """Sets `attrs` as attributes of self._c_func.

    Requires that self._c_func is not None.

    Args:
      attrs: a dictionary from attribute name to attribute proto value
    """
        for name, attr_value in attrs.items():
            serialized = attr_value.SerializeToString()
            with self._c_func.get() as func:
                c_api.TF_FunctionSetAttrValueProto(func, compat.as_str(name), serialized)

    def _create_hash_str(self, input_arg, output_arg, node_def):
        """Creates an 8-character string unique to this input.

    Args:
      input_arg: the input_arg field of an OpDef
                 (e.g. self._definition.signature.input_arg)
      output_arg: the output_arg field of an OpDef
                 (e.g. self._definition.signature.output_arg)
      node_def: the node_def field of a FunctionDef
                (e.g. self._definition.node_def)

    Returns:
      The unique string for this input
    """
        hasher = hashlib.sha1()

        def update_num(n):
            hasher.update(compat.as_bytes('%x' % n))

        def update_str(s):
            update_num(len(s))
            hasher.update(compat.as_bytes(s))

        def update_strs(slist):
            update_num(len(slist))
            for s in slist:
                update_str(s)
        for adef in input_arg:
            update_str(adef.SerializeToString())
        for adef in output_arg:
            update_str(adef.SerializeToString())
        for n in sorted(node_def, key=lambda n: n.name):
            update_str(n.name)
            update_str(n.op)
            update_strs(n.input)
            update_num(len(n.attr))
            for k in sorted(n.attr):
                update_str(k)
                update_str(n.attr[k].SerializeToString())
        return hasher.hexdigest()[:8]

    def add_to_graph(self, g):
        """Adds this function into the graph g."""
        self._create_definition_if_needed()
        if context.executing_eagerly():
            context.context().add_function_def(self.definition)
        else:
            g._add_function(self)
        for f in self._sub_functions.values():
            g._add_function_recursive(f)
        if self._grad_func:
            self._grad_func.add_to_graph(g)

    def __call__(self, *args, **kwargs):
        self.add_to_graph(ops.get_default_graph())
        args = [ops.convert_to_tensor(_) for _ in args] + self._extra_inputs
        ret, op = _call(self._signature, *args, **kwargs)
        assert isinstance(op, ops.Operation)
        setattr(op, '__defun', self)
        if self._shape_func is not None:
            shapes = self._shape_func(op)
            if len(shapes) != len(op.outputs):
                raise ValueError(f'shape_func {self._shape_func} produced {len(shapes):d} shapes, which does not match {len(op.outputs)} outputs.')
            for t, shape in zip(op.outputs, shapes):
                t.set_shape(shape)
        return ret