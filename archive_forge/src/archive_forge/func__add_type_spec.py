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
def _add_type_spec(cls):
    """Creates a nested TypeSpec class for tf.ExtensionType subclass `cls`."""
    spec_name = cls.__name__ + '.Spec'
    spec_qualname = cls.__qualname__ + '.Spec'
    spec_dict = {'value_type': cls, '__module__': cls.__module__}
    user_spec = cls.__dict__.get('Spec', None)
    if user_spec is not None:
        for name, value in user_spec.__dict__.items():
            if extension_type_field.ExtensionTypeField.is_reserved_name(name):
                raise ValueError(f"TypeSpec {spec_qualname} uses reserved name '{name}'.")
            if cls._tf_extension_type_has_field(name):
                raise ValueError(f"TypeSpec {spec_qualname} defines a variable '{name}' which shadows a field in {cls.__qualname__}")
            if name in ('__module__', '__dict__', '__weakref__'):
                continue
            spec_dict[name] = value
    if issubclass(cls, BatchableExtensionType):
        type_spec_base = BatchableExtensionTypeSpec
        if hasattr(cls, '__batch_encoder__') and '__batch_encoder__' not in spec_dict:
            spec_dict['__batch_encoder__'] = cls.__batch_encoder__
    else:
        type_spec_base = ExtensionTypeSpec
        if hasattr(cls, '__batch_encoder__') or '__batch_encoder__' in spec_dict:
            raise ValueError('__batch_encoder__ should only be defined for BatchableExtensionType classes.')
    spec = type(spec_name, (type_spec_base,), spec_dict)
    spec.__qualname__ = spec_qualname
    setattr(cls, 'Spec', spec)
    if '__init__' in spec.__dict__:
        _wrap_user_constructor(spec)
    else:
        _build_spec_constructor(spec)
    cls.__abstractmethods__ -= {'_type_spec'}
    if '__name__' in cls.__dict__:
        type_spec_registry.register(cls.__dict__['__name__'] + '.Spec')(spec)