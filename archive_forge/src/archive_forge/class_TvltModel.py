import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
@add_start_docstrings('The bare TVLT Model transformer outputting raw hidden-states without any specific head on top.', TVLT_START_DOCSTRING)
class TvltModel(TvltPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.pixel_embeddings = TvltPixelEmbeddings(config)
        self.audio_embeddings = TvltAudioEmbeddings(config)
        self.encoder = TvltEncoder(config)
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if config.use_mean_pooling:
            self.layernorm = None
        else:
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return (self.pixel_embeddings.patch_embeddings, self.audio_embeddings.patch_embeddings)

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TVLT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TvltModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, audio_values: torch.FloatTensor, pixel_mask: Optional[torch.FloatTensor]=None, audio_mask: Optional[torch.FloatTensor]=None, mask_pixel: bool=False, mask_audio: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], TvltModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import TvltProcessor, TvltModel
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 8
        >>> images = list(np.random.randn(num_frames, 3, 224, 224))
        >>> audio = list(np.random.randn(10000))

        >>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
        >>> model = TvltModel.from_pretrained("ZinengTang/tvlt-base")

        >>> input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

        >>> outputs = model(**input_dict)
        >>> loss = outputs.loss
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pixel_embedding_output, pixel_mask = self.pixel_embeddings(pixel_values, pixel_mask)
        audio_embedding_output, audio_mask = self.audio_embeddings(audio_values, audio_mask)
        pixel_label_masks = None
        pixel_ids_restore = None
        if mask_pixel:
            pixel_mask_noise, pixel_len_keep = generate_pixel_mask_noise(pixel_embedding_output, pixel_mask=pixel_mask, mask_ratio=self.config.pixel_mask_ratio)
            pixel_embedding_output, pixel_mask, pixel_label_masks, pixel_ids_restore = random_masking(pixel_embedding_output, pixel_mask_noise, pixel_len_keep, attention_masks=pixel_mask)
        audio_label_masks = None
        audio_ids_restore = None
        if mask_audio:
            num_freq_patches = self.config.frequency_length // self.config.audio_patch_size[1]
            audio_mask_noise, audio_len_keep = generate_audio_mask_noise(audio_embedding_output, audio_mask=audio_mask, mask_ratio=self.config.audio_mask_ratio, mask_type=self.config.audio_mask_type, freq_len=num_freq_patches)
            audio_embedding_output, audio_mask, audio_label_masks, audio_ids_restore = random_masking(audio_embedding_output, audio_mask_noise, audio_len_keep, attention_masks=audio_mask)
        batch_size = pixel_values.size(0)
        embedding_output = torch.cat([self.cls_embedding.repeat(batch_size, 1, 1), pixel_embedding_output, audio_embedding_output], 1)
        masked_pixel_len = pixel_embedding_output.size(1)
        attention_mask = None
        if pixel_mask is not None and audio_mask is not None:
            attention_mask = torch.cat([pixel_mask[:, :1], pixel_mask, audio_mask], 1)
        input_shape = embedding_output.size()
        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)
        pixel_sequence_output = sequence_output[:, 1:1 + masked_pixel_len]
        audio_sequence_output = sequence_output[:, 1 + masked_pixel_len:]
        if not return_dict:
            return (sequence_output, pixel_sequence_output, audio_sequence_output, pixel_label_masks, audio_label_masks, pixel_ids_restore, audio_ids_restore) + encoder_outputs[1:]
        return TvltModelOutput(last_hidden_state=sequence_output, last_pixel_hidden_state=pixel_sequence_output, last_audio_hidden_state=audio_sequence_output, pixel_label_masks=pixel_label_masks, audio_label_masks=audio_label_masks, pixel_ids_restore=pixel_ids_restore, audio_ids_restore=audio_ids_restore, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)