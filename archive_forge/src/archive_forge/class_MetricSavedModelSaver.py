from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import data_structures
class MetricSavedModelSaver(layer_serialization.LayerSavedModelSaver):
    """Metric serialization."""

    @property
    def object_identifier(self):
        return constants.METRIC_IDENTIFIER

    def _python_properties_internal(self):
        metadata = dict(class_name=generic_utils.get_registered_name(type(self.obj)), name=self.obj.name, dtype=self.obj.dtype)
        metadata.update(layer_serialization.get_serialized(self.obj))
        if self.obj._build_input_shape is not None:
            metadata['build_input_shape'] = self.obj._build_input_shape
        return metadata

    def _get_serialized_attributes_internal(self, unused_serialization_cache):
        return (dict(variables=data_structures.wrap_or_unwrap(self.obj.variables)), dict())