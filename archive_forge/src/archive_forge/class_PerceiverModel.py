import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
@add_start_docstrings('The Perceiver: a scalable, fully attentional architecture.', PERCEIVER_MODEL_START_DOCSTRING)
class PerceiverModel(PerceiverPreTrainedModel):

    def __init__(self, config, decoder=None, input_preprocessor: PreprocessorType=None, output_postprocessor: PostprocessorType=None):
        super().__init__(config)
        self.config = config
        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.embeddings = PerceiverEmbeddings(config)
        self.encoder = PerceiverEncoder(config, kv_dim=input_preprocessor.num_channels if input_preprocessor is not None else config.d_model)
        self.decoder = decoder
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.latents

    def set_input_embeddings(self, value):
        self.embeddings.latents = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format('(batch_size, sequence_length)'))
    @replace_return_docstrings(output_type=PerceiverModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, inputs: torch.FloatTensor, attention_mask: Optional[torch.FloatTensor]=None, subsampled_output_points: Optional[Dict[str, torch.Tensor]]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, PerceiverModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverImageProcessor, PerceiverModel
        >>> from transformers.models.perceiver.modeling_perceiver import (
        ...     PerceiverTextPreprocessor,
        ...     PerceiverImagePreprocessor,
        ...     PerceiverClassificationDecoder,
        ... )
        >>> import torch
        >>> import requests
        >>> from PIL import Image

        >>> # EXAMPLE 1: using the Perceiver to classify texts
        >>> # - we define a TextPreprocessor, which can be used to embed tokens
        >>> # - we define a ClassificationDecoder, which can be used to decode the
        >>> # final hidden states of the latents to classification logits
        >>> # using trainable position embeddings
        >>> config = PerceiverConfig()
        >>> preprocessor = PerceiverTextPreprocessor(config)
        >>> decoder = PerceiverClassificationDecoder(
        ...     config,
        ...     num_channels=config.d_latents,
        ...     trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
        ...     use_query_residual=True,
        ... )
        >>> model = PerceiverModel(config, input_preprocessor=preprocessor, decoder=decoder)

        >>> # you can then do a forward pass as follows:
        >>> tokenizer = PerceiverTokenizer()
        >>> text = "hello world"
        >>> inputs = tokenizer(text, return_tensors="pt").input_ids

        >>> with torch.no_grad():
        ...     outputs = model(inputs=inputs)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 2]

        >>> # to train, one can train the model using standard cross-entropy:
        >>> criterion = torch.nn.CrossEntropyLoss()

        >>> labels = torch.tensor([1])
        >>> loss = criterion(logits, labels)

        >>> # EXAMPLE 2: using the Perceiver to classify images
        >>> # - we define an ImagePreprocessor, which can be used to embed images
        >>> config = PerceiverConfig(image_size=224)
        >>> preprocessor = PerceiverImagePreprocessor(
        ...     config,
        ...     prep_type="conv1x1",
        ...     spatial_downsample=1,
        ...     out_channels=256,
        ...     position_encoding_type="trainable",
        ...     concat_or_add_pos="concat",
        ...     project_pos_dim=256,
        ...     trainable_position_encoding_kwargs=dict(
        ...         num_channels=256,
        ...         index_dims=config.image_size**2,
        ...     ),
        ... )

        >>> model = PerceiverModel(
        ...     config,
        ...     input_preprocessor=preprocessor,
        ...     decoder=PerceiverClassificationDecoder(
        ...         config,
        ...         num_channels=config.d_latents,
        ...         trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
        ...         use_query_residual=True,
        ...     ),
        ... )

        >>> # you can then do a forward pass as follows:
        >>> image_processor = PerceiverImageProcessor()
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt").pixel_values

        >>> with torch.no_grad():
        ...     outputs = model(inputs=inputs)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 2]

        >>> # to train, one can train the model using standard cross-entropy:
        >>> criterion = torch.nn.CrossEntropyLoss()

        >>> labels = torch.tensor([1])
        >>> loss = criterion(logits, labels)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.input_preprocessor is not None:
            inputs, modality_sizes, inputs_without_pos = self.input_preprocessor(inputs)
        else:
            modality_sizes = None
            inputs_without_pos = None
            if inputs.size()[-1] != self.config.d_model:
                raise ValueError(f"Last dimension of the inputs: {inputs.size()[-1]} doesn't correspond to config.d_model: {self.config.d_model}. Make sure to set config.d_model appropriately.")
        batch_size, seq_length, _ = inputs.size()
        device = inputs.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        extended_attention_mask = self.invert_attention_mask(attention_mask)
        head_mask = self.get_head_mask(head_mask, self.config.num_blocks * self.config.num_self_attends_per_block)
        embedding_output = self.embeddings(batch_size=batch_size)
        encoder_outputs = self.encoder(embedding_output, attention_mask=None, head_mask=head_mask, inputs=inputs, inputs_mask=extended_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        logits = None
        if self.decoder:
            if subsampled_output_points is not None:
                output_modality_sizes = {'audio': subsampled_output_points['audio'].shape[0], 'image': subsampled_output_points['image'].shape[0], 'label': 1}
            else:
                output_modality_sizes = modality_sizes
            decoder_query = self.decoder.decoder_query(inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_output_points)
            decoder_outputs = self.decoder(decoder_query, z=sequence_output, query_mask=extended_attention_mask, output_attentions=output_attentions)
            logits = decoder_outputs.logits
            if output_attentions and decoder_outputs.cross_attentions is not None:
                if return_dict:
                    encoder_outputs.cross_attentions = encoder_outputs.cross_attentions + decoder_outputs.cross_attentions
                else:
                    encoder_outputs = encoder_outputs + decoder_outputs.cross_attentions
            if self.output_postprocessor:
                logits = self.output_postprocessor(logits, modality_sizes=output_modality_sizes)
        if not return_dict:
            if logits is not None:
                return (logits, sequence_output) + encoder_outputs[1:]
            else:
                return (sequence_output,) + encoder_outputs[1:]
        return PerceiverModelOutput(logits=logits, last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)