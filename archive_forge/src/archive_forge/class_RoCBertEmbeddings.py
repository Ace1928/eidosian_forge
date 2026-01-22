import math
import os
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_roc_bert import RoCBertConfig
class RoCBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position, shape, pronunciation and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pronunciation_embed = nn.Embedding(config.pronunciation_vocab_size, config.pronunciation_embed_dim, padding_idx=config.pad_token_id)
        self.shape_embed = nn.Embedding(config.shape_vocab_size, config.shape_embed_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.enable_pronunciation = config.enable_pronunciation
        self.enable_shape = config.enable_shape
        if config.concat_input:
            input_dim = config.hidden_size
            if self.enable_pronunciation:
                pronunciation_dim = config.pronunciation_embed_dim
                input_dim += pronunciation_dim
            if self.enable_shape:
                shape_dim = config.shape_embed_dim
                input_dim += shape_dim
            self.map_inputs_layer = torch.nn.Linear(input_dim, config.hidden_size)
        else:
            self.map_inputs_layer = None
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.register_buffer('token_type_ids', torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device), persistent=False)

    def forward(self, input_ids=None, input_shape_ids=None, input_pronunciation_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if self.map_inputs_layer is None:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = inputs_embeds + token_type_embeddings
            if self.position_embedding_type == 'absolute':
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            denominator = 1
            embedding_in = torch.clone(embeddings)
            if self.enable_shape and input_shape_ids is not None:
                embedding_shape = self.shape_embed(input_shape_ids)
                embedding_in += embedding_shape
                denominator += 1
            if self.enable_pronunciation and input_pronunciation_ids is not None:
                embedding_pronunciation = self.pronunciation_embed(input_pronunciation_ids)
                embedding_in += embedding_pronunciation
                denominator += 1
            embedding_in /= denominator
            return embedding_in
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            device = inputs_embeds.device
            embedding_in = torch.clone(inputs_embeds)
            if self.enable_shape:
                if input_shape_ids is None:
                    input_shape_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
                embedding_shape = self.shape_embed(input_shape_ids)
                embedding_in = torch.cat((embedding_in, embedding_shape), -1)
            if self.enable_pronunciation:
                if input_pronunciation_ids is None:
                    input_pronunciation_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
                embedding_pronunciation = self.pronunciation_embed(input_pronunciation_ids)
                embedding_in = torch.cat((embedding_in, embedding_pronunciation), -1)
            embedding_in = self.map_inputs_layer(embedding_in)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embedding_in += token_type_embeddings
            if self.position_embedding_type == 'absolute':
                position_embeddings = self.position_embeddings(position_ids)
                embedding_in += position_embeddings
            embedding_in = self.LayerNorm(embedding_in)
            embedding_in = self.dropout(embedding_in)
            return embedding_in