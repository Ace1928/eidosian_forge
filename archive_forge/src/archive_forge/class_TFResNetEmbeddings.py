from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig
class TFResNetEmbeddings(keras.layers.Layer):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedder = TFResNetConvLayer(config.num_channels, config.embedding_size, kernel_size=7, stride=2, activation=config.hidden_act, name='embedder')
        self.pooler = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', name='pooler')
        self.num_channels = config.num_channels

    def call(self, pixel_values: tf.Tensor, training: bool=False) -> tf.Tensor:
        _, _, _, num_channels = shape_list(pixel_values)
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        hidden_state = pixel_values
        hidden_state = self.embedder(hidden_state)
        hidden_state = tf.pad(hidden_state, [[0, 0], [1, 1], [1, 1], [0, 0]])
        hidden_state = self.pooler(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embedder', None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)
        if getattr(self, 'pooler', None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)