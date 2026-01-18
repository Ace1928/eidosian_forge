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
callback for `nest.map_structure()`