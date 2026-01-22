from __future__ import annotations
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_swin import SwinConfig
@keras_serializable
class TFSwinMainLayer(keras.layers.Layer):
    config_class = SwinConfig

    def __init__(self, config: SwinConfig, add_pooling_layer: bool=True, use_mask_token: bool=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.embeddings = TFSwinEmbeddings(config, use_mask_token=use_mask_token, name='embeddings')
        self.encoder = TFSwinEncoder(config, self.embeddings.patch_grid, name='encoder')
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.pooler = AdaptiveAveragePooling1D(output_size=(1,)) if add_pooling_layer else None

    def get_input_embeddings(self) -> TFSwinPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_head_mask(self, head_mask: Optional[Any]) -> List:
        if head_mask is not None:
            raise NotImplementedError
        return [None] * len(self.config.depths)

    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFSwinModelOutput, Tuple[tf.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        head_mask = self.get_head_mask(head_mask)
        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, training=training)
        encoder_outputs = self.encoder(embedding_output, input_dimensions, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output, training=training)
        pooled_output = None
        if self.pooler is not None:
            batch_size, _, num_features = shape_list(sequence_output)
            pooled_output = self.pooler(sequence_output)
            pooled_output = tf.reshape(pooled_output, (batch_size, num_features))
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output
        return TFSwinModelOutput(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, reshaped_hidden_states=encoder_outputs.reshaped_hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embeddings', None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'layernorm', None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.num_features])