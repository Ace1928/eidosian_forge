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
class TFEfficientFormerSelfAttention(keras.layers.Layer):

    def __init__(self, dim: int, key_dim: int, num_heads: int, attention_ratio: int, resolution: int, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.scale = key_dim ** (-0.5)
        self.total_key_dim = key_dim * num_heads
        self.expanded_key_dim = int(attention_ratio * key_dim)
        self.total_expanded_key_dim = int(self.expanded_key_dim * num_heads)
        hidden_size = self.total_expanded_key_dim + self.total_key_dim * 2
        self.qkv = keras.layers.Dense(units=hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='qkv')
        self.projection = keras.layers.Dense(units=dim, kernel_initializer=get_initializer(config.initializer_range), name='projection')
        self.resolution = resolution
        self.dim = dim

    def build(self, input_shape: tf.TensorShape) -> None:
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        num_points = len(points)
        attention_offsets = {}
        idxs = []
        for point_1 in points:
            for point_2 in points:
                offset = (abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.add_weight(shape=(self.num_heads, len(attention_offsets)), initializer=keras.initializers.zeros(), trainable=True, name='attention_biases')
        self.attention_bias_idxs = self.add_weight(shape=(num_points, num_points), trainable=False, dtype=tf.int32, name='attention_bias_idxs')
        self.attention_bias_idxs.assign(tf.reshape(tf.cast(idxs, dtype=tf.int32), (num_points, num_points)))
        if self.built:
            return
        self.built = True
        if getattr(self, 'qkv', None) is not None:
            with tf.name_scope(self.qkv.name):
                self.qkv.build([None, None, self.dim])
        if getattr(self, 'projection', None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, self.total_expanded_key_dim])

    def call(self, hidden_states: tf.Tensor, output_attentions: bool=False, training: bool=False) -> Tuple[tf.Tensor]:
        batch_size, sequence_length, *_ = shape_list(hidden_states)
        qkv = self.qkv(inputs=hidden_states)
        query_layer, key_layer, value_layer = tf.split(tf.reshape(tensor=qkv, shape=(batch_size, sequence_length, self.num_heads, -1)), num_or_size_splits=[self.key_dim, self.key_dim, self.expanded_key_dim], axis=3)
        query_layer = tf.transpose(query_layer, perm=[0, 2, 1, 3])
        key_layer = tf.transpose(key_layer, perm=[0, 2, 1, 3])
        value_layer = tf.transpose(value_layer, perm=[0, 2, 1, 3])
        attention_probs = tf.matmul(query_layer, tf.transpose(key_layer, perm=[0, 1, 3, 2]))
        scale = tf.cast(self.scale, dtype=attention_probs.dtype)
        attention_probs = tf.multiply(attention_probs, scale)
        attention_biases = tf.gather(params=self.attention_biases, indices=self.attention_bias_idxs, axis=1)
        attention_probs = attention_probs + attention_biases
        attention_probs = stable_softmax(logits=attention_probs, axis=-1)
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(tensor=context_layer, shape=(batch_size, sequence_length, self.total_expanded_key_dim))
        context_layer = self.projection(context_layer)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs