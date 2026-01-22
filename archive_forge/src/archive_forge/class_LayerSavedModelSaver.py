from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import base_serialization
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
class LayerSavedModelSaver(base_serialization.SavedModelSaver):
    """Implements Layer SavedModel serialization."""

    @property
    def object_identifier(self):
        return constants.LAYER_IDENTIFIER

    @property
    def python_properties(self):
        return self._python_properties_internal()

    def _python_properties_internal(self):
        """Returns dictionary of all python properties."""
        metadata = dict(name=self.obj.name, trainable=self.obj.trainable, expects_training_arg=self.obj._expects_training_arg, dtype=policy.serialize(self.obj._dtype_policy), batch_input_shape=getattr(self.obj, '_batch_input_shape', None), stateful=self.obj.stateful, must_restore_from_config=self.obj._must_restore_from_config)
        metadata.update(get_serialized(self.obj))
        if self.obj.input_spec is not None:
            metadata['input_spec'] = nest.map_structure(lambda x: generic_utils.serialize_keras_object(x) if x else None, self.obj.input_spec)
        if self.obj.activity_regularizer is not None and hasattr(self.obj.activity_regularizer, 'get_config'):
            metadata['activity_regularizer'] = generic_utils.serialize_keras_object(self.obj.activity_regularizer)
        if self.obj._build_input_shape is not None:
            metadata['build_input_shape'] = self.obj._build_input_shape
        return metadata

    def objects_to_serialize(self, serialization_cache):
        return self._get_serialized_attributes(serialization_cache).objects_to_serialize

    def functions_to_serialize(self, serialization_cache):
        return self._get_serialized_attributes(serialization_cache).functions_to_serialize

    def _get_serialized_attributes(self, serialization_cache):
        """Generates or retrieves serialized attributes from cache."""
        keras_cache = serialization_cache.setdefault(constants.KERAS_CACHE_KEY, {})
        if self.obj in keras_cache:
            return keras_cache[self.obj]
        serialized_attr = keras_cache[self.obj] = serialized_attributes.SerializedAttributes.new(self.obj)
        if save_impl.should_skip_serialization(self.obj) or self.obj._must_restore_from_config:
            return serialized_attr
        object_dict, function_dict = self._get_serialized_attributes_internal(serialization_cache)
        serialized_attr.set_and_validate_objects(object_dict)
        serialized_attr.set_and_validate_functions(function_dict)
        return serialized_attr

    def _get_serialized_attributes_internal(self, serialization_cache):
        """Returns dictionary of serialized attributes."""
        objects = save_impl.wrap_layer_objects(self.obj, serialization_cache)
        functions = save_impl.wrap_layer_functions(self.obj, serialization_cache)
        functions['_default_save_signature'] = None
        return (objects, functions)