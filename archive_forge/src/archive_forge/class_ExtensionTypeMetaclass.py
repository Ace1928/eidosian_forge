import abc
import typing
import warnings
import typing_extensions
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
class ExtensionTypeMetaclass(abc.ABCMeta):
    """Metaclass for tf.ExtensionType types."""

    def __init__(cls, name, bases, namespace):
        if not namespace.get('_tf_extension_type_do_not_transform_this_class', False):
            _check_field_annotations(cls)
            _add_extension_type_constructor(cls)
            _add_type_spec(cls)
        super(ExtensionTypeMetaclass, cls).__init__(name, bases, namespace)