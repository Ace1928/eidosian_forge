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
class TFRelPartialLearnableDecoderLayer(keras.layers.Layer):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt=0.0, pre_lnorm=False, r_w_bias=None, r_r_bias=None, layer_norm_epsilon=1e-05, init_std=0.02, output_attentions=False, **kwargs):
        super().__init__(**kwargs)
        self.dec_attn = TFRelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm, r_w_bias=r_w_bias, r_r_bias=r_r_bias, init_std=init_std, layer_norm_epsilon=layer_norm_epsilon, output_attentions=output_attentions, name='dec_attn')
        self.pos_ff = TFPositionwiseFF(d_model, d_inner, dropout, pre_lnorm=pre_lnorm, init_std=init_std, layer_norm_epsilon=layer_norm_epsilon, name='pos_ff')

    def call(self, dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=False):
        attn_outputs = self.dec_attn(dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=training)
        ff_output = self.pos_ff(attn_outputs[0], training=training)
        outputs = [ff_output] + attn_outputs[1:]
        return outputs