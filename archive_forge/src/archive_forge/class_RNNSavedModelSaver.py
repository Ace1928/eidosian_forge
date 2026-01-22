from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import base_serialization
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
class RNNSavedModelSaver(LayerSavedModelSaver):
    """RNN layer serialization."""

    @property
    def object_identifier(self):
        return constants.RNN_LAYER_IDENTIFIER

    def _get_serialized_attributes_internal(self, serialization_cache):
        objects, functions = super(RNNSavedModelSaver, self)._get_serialized_attributes_internal(serialization_cache)
        states = data_structures.wrap_or_unwrap(self.obj.states)
        if isinstance(states, tuple):
            states = data_structures.wrap_or_unwrap(list(states))
        objects['states'] = states
        return (objects, functions)