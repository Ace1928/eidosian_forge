import tree
from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.utils import backend_utils
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
class Cross:

    def __init__(self, feature_names, crossing_dim, output_mode='one_hot'):
        if output_mode not in {'int', 'one_hot'}:
            raise ValueError(f"Invalid value for argument `output_mode`. Expected one of {{'int', 'one_hot'}}. Received: output_mode={output_mode}")
        self.feature_names = tuple(feature_names)
        self.crossing_dim = crossing_dim
        self.output_mode = output_mode

    @property
    def name(self):
        return '_X_'.join(self.feature_names)

    def get_config(self):
        return {'feature_names': self.feature_names, 'crossing_dim': self.crossing_dim, 'output_mode': self.output_mode}

    @classmethod
    def from_config(cls, config):
        return cls(**config)