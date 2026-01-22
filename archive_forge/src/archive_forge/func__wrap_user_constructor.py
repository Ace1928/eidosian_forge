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
def _wrap_user_constructor(cls):
    """Wraps a user-defined constructor for tf.ExtensionType subclass `cls`."""
    user_constructor = cls.__init__

    def wrapped_init(self, *args, **kwargs):
        self.__dict__[_IN_CONSTRUCTOR] = True
        user_constructor(self, *args, **kwargs)
        del self.__dict__[_IN_CONSTRUCTOR]
        self._tf_extension_type_convert_fields()
        self.__validate__()
    cls.__init__ = tf_decorator.make_decorator(user_constructor, wrapped_init)