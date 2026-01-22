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
@tf_export('experimental.ExtensionTypeSpec')
class ExtensionTypeSpec(type_spec.TypeSpec):
    """Base class for tf.ExtensionType TypeSpec."""

    def _serialize(self):
        fields = [f.name for f in self._tf_extension_type_fields()]
        if self._tf_extension_type_is_packed:
            fields.append('_tf_extension_type_is_packed')
        return tuple(((f, _change_nested_mappings_to(self.__dict__[f], dict)) for f in fields))

    @classmethod
    def _deserialize(cls, state):
        state = _change_nested_mappings_to(state, immutable_dict.ImmutableDict)
        return _create_object_from_type_and_dict(cls, state)

    def __reduce__(self):
        return (_deserialize_for_reduce, (self.value_type, self._serialize()))

    def _to_components(self, value):
        if self._tf_extension_type_is_packed:
            return value._tf_extension_type_packed_variant
        tensor_or_composite = (tensor.Tensor, composite_tensor.CompositeTensor)
        value_tuple = tuple((value.__dict__[key] for key in self.__dict__))
        return tuple((x for x in nest.flatten(value_tuple) if isinstance(x, tensor_or_composite)))

    def _from_components(self, components):
        if self._tf_extension_type_is_packed:
            return _create_object_from_type_and_dict(self.value_type, {'_tf_extension_type_cached_type_spec': self, '_tf_extension_type_packed_variant': components})
        spec_tuple = tuple(self.__dict__.values())
        components_iter = iter(components)
        flat = [next(components_iter) if isinstance(x, type_spec.TypeSpec) else x for x in nest.flatten(spec_tuple)]
        if list(components_iter):
            raise ValueError('Cannot build an ExtensionType instance from components because more components are provided than the number expected by the type spec.')
        value_tuple = nest.pack_sequence_as(spec_tuple, flat)
        fields = dict(zip(self.__dict__.keys(), value_tuple))
        return _create_object_from_type_and_dict(self.value_type, fields)

    @property
    def _component_specs(self):
        if self._tf_extension_type_is_packed:
            return tensor.TensorSpec((), dtypes.variant)
        components = []

        def push_if_type_spec(x):
            if isinstance(x, type_spec.TypeSpec):
                components.append(x)
        nest.map_structure(push_if_type_spec, tuple(self.__dict__.values()))
        return tuple(components)

    @classmethod
    def from_value(cls, value):
        cached_spec = getattr(value, '_tf_extension_type_cached_type_spec', None)
        if cached_spec is not None:
            return cached_spec
        value_fields = value.__dict__
        spec_fields = nest.map_structure(_replace_tensor_with_spec, value_fields)
        spec_fields.pop('_tf_extension_type_cached_fields', None)
        return _create_object_from_type_and_dict(cls, spec_fields)

    def __setattr__(self, name, value):
        if hasattr(self, _IN_CONSTRUCTOR) and self._tf_extension_type_has_field(name):
            self.__dict__[name] = value
        elif name in type_spec.CACHED_FIXED_PROPERTIES:
            super().__setattr__(name, value)
        else:
            raise AttributeError(f'Cannot mutate attribute `{name}` outside the custom constructor of ExtensionTypeSpec.')

    def __delattr__(self, name):
        if hasattr(self, _IN_CONSTRUCTOR) and self._tf_extension_type_has_field(name):
            del self.__dict__[name]
        else:
            raise AttributeError(f'Cannot mutate attribute `{name}` outside the custom constructor of ExtensionTypeSpec.')

    def __validate__(self):
        """Perform post-construction validation."""

    @classmethod
    def _tf_extension_type_fields(cls):
        return cls.value_type._tf_extension_type_fields()

    @classmethod
    def _tf_extension_type_has_field(cls, name):
        return any((name == field.name for field in cls._tf_extension_type_fields()))

    def _tf_extension_type_convert_fields(self):
        extension_type_field.convert_fields_for_spec(self._tf_extension_type_fields(), self.__dict__)

    def __repr__(self):
        fields = ', '.join([f'{k}={v!r}' for k, v in self._serialize()])
        return f'{type(self).__qualname__}({fields})'
    _tf_extension_type_is_packed = False

    def _tf_extension_type_with_packed(self, value):
        """Returns a copy of this `TypeSpec` with `packed=value`.

    Args:
      value: A boolean value.

    Returns:
      A copy of `self` with `_tf_extension_type_is_packed=value`.
    """
        copy = _create_object_from_type_and_dict(type(self), self.__dict__)
        copy.__dict__['_tf_extension_type_is_packed'] = value
        return copy

    def _to_legacy_output_shapes(self):
        """Returns the shape property."""
        try:
            return self.shape
        except AttributeError as e:
            raise NotImplementedError('It appears that the Spec of the ExtensionType is missing a shape property. In order to support tf.Data, it is recommended that you implement a shape property on the Spec.') from e