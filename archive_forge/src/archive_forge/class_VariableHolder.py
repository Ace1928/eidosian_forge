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
class VariableHolder(object):
    """Holds variables for a python function."""

    def __init__(self, fn=None, share_variables=False):
        self._fn = fn
        self._share_variables = share_variables
        self._variables_by_name = data_structures.Mapping()

    @property
    def variables(self):
        return self._variables_by_name

    def variable_creator_scope(self, next_creator, **kwargs):
        """Creates variables & adds them to collections to match legacy code."""
        collections = kwargs.pop('collections', None)
        v = None
        with ops.name_scope(kwargs.get('name', None), 'Variable', skip_on_eager=False) as name:
            variable_name = ops.name_from_scope_name(name)
            kwargs['name'] = name
        if self._share_variables:
            v = self._variables_by_name.get(variable_name, None)
        if v is None:
            v = next_creator(**kwargs)
            self._variables_by_name[variable_name] = v
        if collections is None:
            collections = [ops.GraphKeys.GLOBAL_VARIABLES]
        if v.trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
            collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
        ops.add_to_collections(collections, v)
        return v

    def __call__(self, *args, **kwargs):
        return self.call_with_variable_creator_scope(self._fn)(*args, **kwargs)

    def call_with_variable_creator_scope(self, fn):

        def wrapped(*args, **kwargs):
            with variable_scope.variable_creator_scope(self.variable_creator_scope):
                return fn(*args, **kwargs)
        return wrapped