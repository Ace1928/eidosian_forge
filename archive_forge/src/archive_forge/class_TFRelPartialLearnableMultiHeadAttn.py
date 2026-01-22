from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ....modeling_tf_utils import (
from ....tf_utils import shape_list, stable_softmax
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask
class TFRelPartialLearnableMultiHeadAttn(keras.layers.Layer):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.0, pre_lnorm=False, r_r_bias=None, r_w_bias=None, layer_norm_epsilon=1e-05, init_std=0.02, output_attentions=False, **kwargs):
        super().__init__(**kwargs)
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.output_attentions = output_attentions
        self.qkv_net = keras.layers.Dense(3 * n_head * d_head, kernel_initializer=get_initializer(init_std), use_bias=False, name='qkv_net')
        self.drop = keras.layers.Dropout(dropout)
        self.dropatt = keras.layers.Dropout(dropatt)
        self.o_net = keras.layers.Dense(d_model, kernel_initializer=get_initializer(init_std), use_bias=False, name='o_net')
        self.layer_norm = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name='layer_norm')
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm
        if r_r_bias is not None and r_w_bias is not None:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        else:
            self.r_r_bias = None
            self.r_w_bias = None
        self.r_net = keras.layers.Dense(self.n_head * self.d_head, kernel_initializer=get_initializer(init_std), use_bias=False, name='r_net')

    def build(self, input_shape):
        if self.r_r_bias is None or self.r_w_bias is None:
            self.r_r_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_r_bias')
            self.r_w_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_w_bias')
        super().build(input_shape)

    def _rel_shift(self, x):
        x_size = shape_list(x)
        x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
        x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_size)
        return x

    def call(self, w, r, attn_mask, mems, head_mask, output_attentions, training=False):
        qlen, rlen, bsz = (shape_list(w)[0], shape_list(r)[0], shape_list(w)[1])
        if mems is not None:
            mems = tf.cast(mems, dtype=w.dtype)
            cat = tf.concat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, axis=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, axis=-1)
        klen = shape_list(w_head_k)[0]
        w_head_q = tf.reshape(w_head_q, (qlen, bsz, self.n_head, self.d_head))
        w_head_k = tf.reshape(w_head_k, (klen, bsz, self.n_head, self.d_head))
        w_head_v = tf.reshape(w_head_v, (klen, bsz, self.n_head, self.d_head))
        r_head_k = tf.reshape(r_head_k, (rlen, self.n_head, self.d_head))
        rw_head_q = w_head_q + self.r_w_bias
        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        rr_head_q = w_head_q + self.r_r_bias
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        attn_score = attn_score * self.scale
        if attn_mask is not None:
            attn_mask_t = attn_mask[:, :, None, None]
            attn_mask_t = tf.cast(attn_mask_t, dtype=attn_score.dtype)
            attn_score = attn_score * (1.0 - attn_mask_t) - 1e+30 * attn_mask_t
        attn_prob = stable_softmax(attn_score, axis=1)
        attn_prob = self.dropatt(attn_prob, training=training)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        attn_vec_sizes = shape_list(attn_vec)
        attn_vec = tf.reshape(attn_vec, (attn_vec_sizes[0], attn_vec_sizes[1], self.n_head * self.d_head))
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out, training=training)
        if self.pre_lnorm:
            outputs = [w + attn_out]
        else:
            outputs = [self.layer_norm(w + attn_out)]
        if output_attentions:
            outputs.append(attn_prob)
        return outputs