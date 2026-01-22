import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class TFEfficientFormerMeta4D(keras.layers.Layer):

    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float=0.0, **kwargs):
        super().__init__(**kwargs)
        pool_size = config.pool_size if config.pool_size is not None else 3
        self.token_mixer = TFEfficientFormerPooling(pool_size=pool_size, name='token_mixer')
        self.dim = dim
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = TFEfficientFormerConvMlp(config=config, in_features=dim, hidden_features=mlp_hidden_dim, drop=config.hidden_dropout_prob, name='mlp')
        self.drop_path = TFEfficientFormerDropPath(drop_path, name='drop_path') if drop_path > 0.0 else keras.layers.Activation('linear', name='drop_path')
        self.config = config

    def build(self, input_shape=None):
        self.layer_scale_1 = None
        self.layer_scale_2 = None
        if self.config.use_layer_scale:
            self.layer_scale_1 = self.add_weight(shape=self.dim, initializer=keras.initializers.Constant(value=self.config.layer_scale_init_value), trainable=True, name='layer_scale_1')
            self.layer_scale_2 = self.add_weight(shape=self.dim, initializer=keras.initializers.Constant(value=self.config.layer_scale_init_value), trainable=True, name='layer_scale_2')
        if self.built:
            return
        self.built = True
        if getattr(self, 'token_mixer', None) is not None:
            with tf.name_scope(self.token_mixer.name):
                self.token_mixer.build(None)
        if getattr(self, 'mlp', None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        if getattr(self, 'drop_path', None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> Tuple[tf.Tensor]:
        outputs = self.token_mixer(hidden_states)
        if self.config.use_layer_scale:
            layer_output = hidden_states + self.drop_path(tf.expand_dims(tf.expand_dims(self.layer_scale_1, 0), 0) * outputs, training=training)
            layer_output = layer_output + self.drop_path(tf.expand_dims(tf.expand_dims(self.layer_scale_2, 0), 0) * self.mlp(hidden_state=layer_output, training=training), training=training)
        else:
            layer_output = hidden_states + self.drop_path(outputs, training=training)
            layer_output = layer_output + self.drop_path(self.mlp(hidden_state=layer_output, training=training), training=training)
        return layer_output