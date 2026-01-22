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
@tf_export('experimental.ExtensionType')
class ExtensionType(composite_tensor.CompositeTensor, metaclass=ExtensionTypeMetaclass):
    """Base class for TensorFlow `ExtensionType` classes.

  Tensorflow `ExtensionType` classes are specialized Python classes that can be
  used transparently with TensorFlow -- e.g., they can be used with ops
  such as `tf.cond` or `tf.while_loop` and used as inputs or outputs for
  `tf.function` and Keras layers.

  New `ExtensionType` classes are defined by creating a subclass of
  `tf.ExtensionType` that
  contains type annotations for all instance variables.  The following type
  annotations are supported:

  Type                      | Example
  ------------------------- | --------------------------------------------
  Python integers           | `i: int`
  Python floats             | `f: float`
  Python strings            | `s: str`
  Python booleans           | `b: bool`
  Python None               | `n: None`
  Python tuple              | `params: tuple[int, float, int, int]`
  Python tuple w/ Ellipsis  | `lengths: tuple[int, ...]`
  Tensors                   | `t: tf.Tensor`
  Composite Tensors         | `rt: tf.RaggedTensor`
  Extension Types           | `m: MyMaskedTensor`
  Tensor shapes             | `shape: tf.TensorShape`
  Tensor dtypes             | `dtype: tf.DType`
  Type unions               | `length: typing.Union[int, float]`
  Tuples                    | `params: typing.Tuple[int, float, int, int]`
  Tuples w/ Ellipsis        | `lengths: typing.Tuple[int, ...]`
  Mappings                  | `tags: typing.Mapping[str, str]`

  Fields annotated with `typing.Mapping` will be stored using an immutable
  mapping type.

  ExtensionType values are immutable -- i.e., once constructed, you can not
  modify or delete any of their instance members.

  ### Examples

  >>> class MaskedTensor(ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor

  >>> class Toy(ExtensionType):
  ...   name: str
  ...   price: tensor.Tensor
  ...   features: typing.Mapping[str, tf.Tensor]

  >>> class ToyStore(ExtensionType):
  ...   name: str
  ...   toys: typing.Tuple[Toy, ...]
  """
    _tf_extension_type_do_not_transform_this_class = True

    def __init__(self, *args, **kwargs):
        if type(self) is ExtensionType:
            raise AssertionError('Cannot create an instance of ExtensionType because ExtensionType is an abstract base class.')
    _tf_extension_type_cached_fields = None

    @classmethod
    def _tf_extension_type_fields(cls):
        """An ordered list describing the fields of this ExtensionType.

    Returns:
      A list of `ExtensionTypeField` objects.  Forward references are resolved
      if possible, or left unresolved otherwise.
    """
        if '_tf_extension_type_cached_fields' in cls.__dict__:
            return cls._tf_extension_type_cached_fields
        try:
            type_hints = typing_extensions.get_type_hints(cls, include_extras=False)
            ok_to_cache = True
        except (NameError, AttributeError):
            type_hints = {}
            for base in reversed(cls.__mro__):
                type_hints.update(base.__dict__.get('__annotations__', {}))
            ok_to_cache = False
        fields = []
        for name, value_type in type_hints.items():
            default = getattr(cls, name, extension_type_field.ExtensionTypeField.NO_DEFAULT)
            fields.append(extension_type_field.ExtensionTypeField(name, value_type, default))
        fields = tuple(fields)
        if ok_to_cache:
            cls._tf_extension_type_cached_fields = fields
        return fields

    @classmethod
    def _tf_extension_type_has_field(cls, name):
        return any((name == field.name for field in cls._tf_extension_type_fields()))

    def _tf_extension_type_convert_fields(self):
        extension_type_field.convert_fields(self._tf_extension_type_fields(), self.__dict__)

    def __repr__(self):
        fields = ', '.join([f'{field.name}={getattr(self, field.name)!r}' for field in self._tf_extension_type_fields()])
        return f'{type(self).__qualname__}({fields})'

    def __setattr__(self, name, value):
        if name in _MUTABLE_KERAS_PROPERTIES or (hasattr(self, _IN_CONSTRUCTOR) and self._tf_extension_type_has_field(name)):
            self.__dict__[name] = value
        else:
            raise AttributeError(f'Cannot mutate attribute `{name}` outside the custom constructor of ExtensionType.')

    def __delattr__(self, name):
        if name in _MUTABLE_KERAS_PROPERTIES or (hasattr(self, _IN_CONSTRUCTOR) and self._tf_extension_type_has_field(name)):
            del self.__dict__[name]
        else:
            raise AttributeError(f'Cannot mutate attribute `{name}` outside the custom constructor of ExtensionType.')

    def __getattr__(self, name):
        if name in _MUTABLE_KERAS_PROPERTIES:
            return object.__getattribute__(self, name)
        if '_tf_extension_type_packed_variant' in self.__dict__:
            return getattr(unpack(self), name)
        raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        if self._type_spec != other._type_spec:
            return False
        self_tensors = nest.flatten(self, expand_composites=True)
        other_tensors = nest.flatten(other, expand_composites=True)
        if len(self_tensors) != len(other_tensors):
            return False
        conditions = []
        for t1, t2 in zip(self_tensors, other_tensors):
            conditions.append(math_ops.reduce_all(gen_math_ops.equal(array_ops.shape(t1), array_ops.shape(t2), incompatible_shape_error=False)))
            conditions.append(math_ops.reduce_all(gen_math_ops.equal(t1, t2, incompatible_shape_error=False)))
        return math_ops.reduce_all(array_ops_stack.stack(conditions))

    def __ne__(self, other):
        eq = self.__eq__(other)
        if isinstance(eq, tensor.Tensor):
            return math_ops.logical_not(eq)
        else:
            return not eq

    def __validate__(self):
        """Perform post-construction validation."""
    _tf_extension_type_cached_type_spec = None

    @property
    def _type_spec(self):
        if self._tf_extension_type_cached_type_spec is None:
            assert not is_packed(self)
            self.__dict__['_tf_extension_type_cached_type_spec'] = self.Spec.from_value(self)
        return self._tf_extension_type_cached_type_spec