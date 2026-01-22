from tensorflow.python.framework.device import DeviceSpec
from tensorflow.python.framework.indexed_slices import IndexedSlices
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework.ops import Operation
from tensorflow.python.framework.tensor import Tensor
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.framework.sparse_tensor import SparseTensorValue
from tensorflow.python.framework.indexed_slices import convert_to_tensor_or_indexed_slices
from tensorflow.python.framework.ops import device
from tensorflow.python.framework.ops import container
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.framework.ops import op_scope
from tensorflow.python.framework.ops import colocate_with
from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.framework.ops import get_default_graph
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import add_to_collection
from tensorflow.python.framework.ops import add_to_collections
from tensorflow.python.framework.ops import get_collection
from tensorflow.python.framework.ops import get_collection_ref
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework.random_seed import get_seed
from tensorflow.python.framework.random_seed import set_random_seed
from tensorflow.python.framework.sparse_tensor import convert_to_tensor_or_sparse_tensor
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.framework.tensor_util import make_tensor_proto
from tensorflow.python.framework.tensor_util import MakeNdarray as make_ndarray
from tensorflow.python.framework.ops import RegisterGradient
from tensorflow.python.framework.ops import NotDifferentiable
from tensorflow.python.framework.ops import NoGradient
from tensorflow.python.framework.tensor_shape import Dimension
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.tensor_conversion_registry import register_tensor_conversion_function
from tensorflow.python.framework.dtypes import *  # pylint: disable=redefined-builtin
from tensorflow.python.framework.load_library import *
Classes and functions for building TensorFlow graphs.