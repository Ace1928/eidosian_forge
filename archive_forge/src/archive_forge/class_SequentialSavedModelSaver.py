from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.saving.saved_model import save_impl
class SequentialSavedModelSaver(ModelSavedModelSaver):

    @property
    def object_identifier(self):
        return constants.SEQUENTIAL_IDENTIFIER