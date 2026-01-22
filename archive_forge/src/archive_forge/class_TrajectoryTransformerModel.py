import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_trajectory_transformer import TrajectoryTransformerConfig
@add_start_docstrings('The bare TrajectoryTransformer Model transformer outputting raw hidden-states without any specific head on top.', TRAJECTORY_TRANSFORMER_START_DOCSTRING)
class TrajectoryTransformerModel(TrajectoryTransformerPreTrainedModel):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size * config.transition_dim + 1, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = EinLinear(config.transition_dim, config.n_embd, config.vocab_size + 1, bias=False)
        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.embedding_dim = config.n_embd
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight
        self.gradient_checkpointing = False
        self.post_init()

    def get_block_size(self):
        return self.block_size

    def offset_tokens(self, trajectories):
        _, sequence_length = trajectories.shape
        n_states = int(np.ceil(sequence_length / self.transition_dim))
        offsets = torch.arange(self.transition_dim) * self.vocab_size
        offsets = offsets.repeat(n_states).to(trajectories.device)
        offset_trajectories = trajectories + offsets[:sequence_length]
        offset_trajectories[trajectories == self.vocab_size] = self.stop_token
        return offset_trajectories

    def pad_to_full_observation(self, hidden_states):
        batch_size, sequence_length, _ = hidden_states.shape
        n_pad = (self.transition_dim - sequence_length % self.transition_dim) % self.transition_dim
        padding = torch.zeros(batch_size, n_pad, self.embedding_dim, device=hidden_states.device)
        hidden_states_pad = torch.cat([hidden_states, padding], dim=1)
        hidden_states_pad = hidden_states_pad.view(-1, self.transition_dim, self.embedding_dim)
        return (hidden_states_pad, n_pad)

    @add_start_docstrings_to_model_forward(TRAJECTORY_TRANSFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TrajectoryTransformerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, trajectories: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, targets: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], TrajectoryTransformerOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import TrajectoryTransformerModel
        >>> import torch

        >>> model = TrajectoryTransformerModel.from_pretrained(
        ...     "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"
        ... )
        >>> model.to(device)
        >>> model.eval()

        >>> observations_dim, action_dim, batch_size = 17, 6, 256
        >>> seq_length = observations_dim + action_dim + 1

        >>> trajectories = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(
        ...     device
        ... )
        >>> targets = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(device)

        >>> outputs = model(
        ...     trajectories,
        ...     targets=targets,
        ...     use_cache=True,
        ...     output_attentions=True,
        ...     output_hidden_states=True,
        ...     return_dict=True,
        ... )
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.blocks))
        batch_size, sequence_length = trajectories.size()
        if sequence_length > self.block_size:
            raise ValueError('Cannot forward, model block size is exhausted.')
        offset_trajectories = self.offset_tokens(trajectories)
        token_embeddings = self.tok_emb(offset_trajectories)
        position_embeddings = self.pos_emb[:, :sequence_length, :]
        hidden_states = self.drop(token_embeddings + position_embeddings)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, layer_past, use_cache, output_attentions)
            else:
                outputs = block(hidden_states, layer_past, use_cache, output_attentions)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
        hidden_state = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states_pad, n_pad = self.pad_to_full_observation(hidden_state)
        logits = self.head(hidden_states_pad)
        logits = logits.reshape(batch_size, sequence_length + n_pad, self.vocab_size + 1)
        logits = logits[:, :sequence_length]
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction='none')
            if self.action_weight != 1 or self.reward_weight != 1 or self.value_weight != 1:
                n_states = int(np.ceil(sequence_length / self.transition_dim))
                weights = torch.cat([torch.ones(self.observation_dim, device=trajectories.device), torch.ones(self.action_dim, device=trajectories.device) * self.action_weight, torch.ones(1, device=trajectories.device) * self.reward_weight, torch.ones(1, device=trajectories.device) * self.value_weight])
                weights = weights.repeat(n_states)
                weights = weights[1:].repeat(batch_size, 1)
                loss = loss * weights.view(-1)
            loss = (loss * attention_mask.view(-1)).mean()
        else:
            loss = None
        if not return_dict:
            return tuple((v for v in [loss, logits, presents, all_hidden_states, all_self_attentions] if v is not None))
        return TrajectoryTransformerOutput(loss=loss, logits=logits, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attentions)