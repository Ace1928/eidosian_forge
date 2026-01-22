import collections
import copy
import os
import keras_tuner
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.layers.experimental import preprocessing
from autokeras import pipeline as pipeline_module
from autokeras.utils import data_utils
from autokeras.utils import utils
def _prepare_model_build(self, hp, **kwargs):
    """Prepare for building the Keras model.

        It builds the Pipeline from HyperPipeline, transforms the dataset to set
        the input shapes and output shapes of the HyperModel.
        """
    dataset = kwargs['x']
    pipeline = self.hyper_pipeline.build(hp, dataset)
    pipeline.fit(dataset)
    dataset = pipeline.transform(dataset)
    self.hypermodel.set_io_shapes(data_utils.dataset_shape(dataset))
    if 'validation_data' in kwargs:
        validation_data = pipeline.transform(kwargs['validation_data'])
    else:
        validation_data = None
    return (pipeline, dataset, validation_data)