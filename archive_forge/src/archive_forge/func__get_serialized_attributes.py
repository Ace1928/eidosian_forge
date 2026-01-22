from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import base_serialization
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
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