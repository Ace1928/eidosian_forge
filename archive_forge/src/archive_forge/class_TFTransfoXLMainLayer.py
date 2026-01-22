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
@keras_serializable
class TFTransfoXLMainLayer(keras.layers.Layer):
    config_class = TransfoXLConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.use_return_dict
        self.n_token = config.vocab_size
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.untie_r = config.untie_r
        self.word_emb = TFAdaptiveEmbedding(config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val, init_std=config.init_std, name='word_emb')
        self.drop = keras.layers.Dropout(config.dropout)
        self.n_layer = config.n_layer
        self.mem_len = config.mem_len
        self.attn_type = config.attn_type
        self.layers = []
        if config.attn_type == 0:
            for i in range(config.n_layer):
                self.layers.append(TFRelPartialLearnableDecoderLayer(config.n_head, config.d_model, config.d_head, config.d_inner, config.dropout, dropatt=config.dropatt, pre_lnorm=config.pre_lnorm, r_w_bias=None if self.untie_r else self.r_w_bias, r_r_bias=None if self.untie_r else self.r_r_bias, layer_norm_epsilon=config.layer_norm_epsilon, init_std=config.init_std, output_attentions=self.output_attentions, name=f'layers_._{i}'))
        else:
            raise NotImplementedError
        self.same_length = config.same_length
        self.clamp_len = config.clamp_len
        if self.attn_type == 0:
            self.pos_emb = TFPositionalEmbedding(self.d_model, name='pos_emb')
        else:
            raise NotImplementedError

    def build(self, input_shape):
        if not self.untie_r:
            self.r_w_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_w_bias')
            self.r_r_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_r_bias')
        super().build(input_shape)

    def get_input_embeddings(self):
        return self.word_emb

    def set_input_embeddings(self, value):
        raise NotImplementedError

    def backward_compatible(self):
        self.sample_softmax = -1

    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    def _prune_heads(self, heads):
        raise NotImplementedError

    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            for i in range(self.n_layer):
                empty = tf.zeros([self.mem_len, bsz, self.d_model])
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        if mems is None:
            return None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'
        new_mems = []
        end_idx = mlen + tf.math.maximum(0, qlen)
        beg_idx = tf.math.maximum(0, end_idx - tf.convert_to_tensor(self.mem_len))
        for i in range(len(hids)):
            mems[i] = tf.cast(mems[i], dtype=hids[i].dtype)
            cat = tf.concat([mems[i], hids[i]], axis=0)
            tf.stop_gradient(cat)
            new_mems.append(cat[beg_idx:end_idx])
        return new_mems

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, mems: List[tf.Tensor] | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_ids = tf.transpose(input_ids, perm=(1, 0))
            qlen, bsz = shape_list(input_ids)
        elif inputs_embeds is not None:
            inputs_embeds = tf.transpose(inputs_embeds, perm=(1, 0, 2))
            qlen, bsz = shape_list(inputs_embeds)[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if mems is None:
            mems = self.init_mems(bsz)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.n_layer
        if inputs_embeds is not None:
            word_emb = inputs_embeds
        else:
            word_emb = self.word_emb(input_ids)
        mlen = shape_list(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen
        all_ones = tf.ones([qlen, klen], dtype=tf.int32)
        upper_mask = 1 - tf.linalg.band_part(tf.ones([qlen, klen], dtype=tf.int32), -1, mlen)
        if self.same_length:
            mask_len = klen - self.mem_len
            mask_shift_len = qlen - tf.nn.relu(mask_len)
            lower_mask = tf.linalg.band_part(all_ones, -1, 0) - tf.linalg.band_part(all_ones, mask_shift_len - 1, 0) * tf.cast(mask_shift_len != 0, tf.int32)
            dec_attn_mask = upper_mask + lower_mask
        else:
            dec_attn_mask = upper_mask
        hids = []
        attentions = [] if output_attentions else None
        if self.attn_type == 0:
            pos_seq = tf.range(klen - 1, -1, -1.0)
            if self.clamp_len > 0:
                pos_seq = tf.minimum(pos_seq, self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            core_out = self.drop(word_emb, training=training)
            pos_emb = self.drop(pos_emb, training=training)
            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                mems_i = None if mems is None else mems[i]
                layer_outputs = layer(core_out, pos_emb, dec_attn_mask, mems_i, head_mask[i], output_attentions, training=training)
                core_out = layer_outputs[0]
                if output_attentions:
                    attentions.append(layer_outputs[1])
        else:
            raise NotImplementedError
        core_out = self.drop(core_out, training=training)
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        core_out = tf.transpose(core_out, perm=(1, 0, 2))
        if output_hidden_states:
            hids = tuple((tf.transpose(t, perm=(1, 0, 2)) for t in hids))
            hids = hids + (core_out,)
        else:
            hids = None
        if output_attentions:
            attentions = tuple((tf.transpose(t, perm=(2, 3, 0, 1)) for t in attentions))
        if not return_dict:
            return tuple((v for v in [core_out, new_mems, hids, attentions] if v is not None))
        return TFTransfoXLModelOutput(last_hidden_state=core_out, mems=new_mems, hidden_states=hids, attentions=attentions)