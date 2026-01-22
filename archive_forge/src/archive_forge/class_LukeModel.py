import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_luke import LukeConfig
@add_start_docstrings('The bare LUKE model transformer outputting raw hidden-states for both word tokens and entities without any specific head on top.', LUKE_START_DOCSTRING)
class LukeModel(LukePreTrainedModel):

    def __init__(self, config: LukeConfig, add_pooling_layer: bool=True):
        super().__init__(config)
        self.config = config
        self.embeddings = LukeEmbeddings(config)
        self.entity_embeddings = LukeEntityEmbeddings(config)
        self.encoder = LukeEncoder(config)
        self.pooler = LukePooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_entity_embeddings(self):
        return self.entity_embeddings.entity_embeddings

    def set_entity_embeddings(self, value):
        self.entity_embeddings.entity_embeddings = value

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError('LUKE does not support the pruning of attention heads')

    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=BaseLukeModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, entity_ids: Optional[torch.LongTensor]=None, entity_attention_mask: Optional[torch.FloatTensor]=None, entity_token_type_ids: Optional[torch.LongTensor]=None, entity_position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseLukeModelOutputWithPooling]:
        """

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LukeModel

        >>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
        >>> model = LukeModel.from_pretrained("studio-ousia/luke-base")
        # Compute the contextualized entity representation corresponding to the entity mention "Beyoncé"

        >>> text = "Beyoncé lives in Los Angeles."
        >>> entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"

        >>> encoding = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
        >>> outputs = model(**encoding)
        >>> word_last_hidden_state = outputs.last_hidden_state
        >>> entity_last_hidden_state = outputs.entity_last_hidden_state
        # Input Wikipedia entities to obtain enriched contextualized representations of word tokens

        >>> text = "Beyoncé lives in Los Angeles."
        >>> entities = [
        ...     "Beyoncé",
        ...     "Los Angeles",
        ... ]  # Wikipedia entity titles corresponding to the entity mentions "Beyoncé" and "Los Angeles"
        >>> entity_spans = [
        ...     (0, 7),
        ...     (17, 28),
        ... ]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"

        >>> encoding = tokenizer(
        ...     text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt"
        ... )
        >>> outputs = model(**encoding)
        >>> word_last_hidden_state = outputs.last_hidden_state
        >>> entity_last_hidden_state = outputs.entity_last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if entity_ids is not None:
            entity_seq_length = entity_ids.size(1)
            if entity_attention_mask is None:
                entity_attention_mask = torch.ones((batch_size, entity_seq_length), device=device)
            if entity_token_type_ids is None:
                entity_token_type_ids = torch.zeros((batch_size, entity_seq_length), dtype=torch.long, device=device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        word_embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, entity_attention_mask)
        if entity_ids is None:
            entity_embedding_output = None
        else:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_token_type_ids)
        encoder_outputs = self.encoder(word_embedding_output, entity_embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseLukeModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, entity_last_hidden_state=encoder_outputs.entity_last_hidden_state, entity_hidden_states=encoder_outputs.entity_hidden_states)

    def get_extended_attention_mask(self, word_attention_mask: torch.LongTensor, entity_attention_mask: Optional[torch.LongTensor]):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            word_attention_mask (`torch.LongTensor`):
                Attention mask for word tokens with ones indicating tokens to attend to, zeros for tokens to ignore.
            entity_attention_mask (`torch.LongTensor`, *optional*):
                Attention mask for entity tokens with ones indicating tokens to attend to, zeros for tokens to ignore.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=-1)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f'Wrong shape for attention_mask (shape {attention_mask.shape})')
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        return extended_attention_mask