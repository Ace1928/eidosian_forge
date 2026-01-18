import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import preprocessors as preprocessors_module
from autokeras.engine import hyper_preprocessor as hpps_module
from autokeras.engine import preprocessor as pps_module
from autokeras.utils import data_utils
from autokeras.utils import io_utils
def transform_y(self, dataset):
    """Transform the target dataset for the model.

        # Arguments
            dataset: tf.data.Dataset. The target dataset for the model.

        # Returns
            An instance of tf.data.Dataset. The transformed dataset.
        """
    return self._transform_data(dataset, self.outputs)