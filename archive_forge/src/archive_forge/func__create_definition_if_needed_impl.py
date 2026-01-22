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