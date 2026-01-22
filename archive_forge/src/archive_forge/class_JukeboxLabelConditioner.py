import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
class JukeboxLabelConditioner(nn.Module):

    def __init__(self, config, include_time_signal):
        super().__init__()
        embed_dim = config.hidden_size
        timing_dims = config.timing_dims
        sampling_rate = config.sampling_rate
        nb_genres, nb_artists = config.metadata_dims
        music_tokens_shape = config.n_ctx
        self.max_nb_genres = config.max_nb_genres
        self.bow_genre_emb = nn.Embedding(nb_genres, embed_dim)
        self.artist_emb = nn.Embedding(nb_artists, embed_dim)
        self.include_time_signal = include_time_signal
        if self.include_time_signal:
            total_length_range = (config.min_duration * sampling_rate, config.max_duration * sampling_rate)
            absolute_pos_range = (0.0, config.max_duration * sampling_rate)
            relative_pos_range = (0.0, 1.0)
            self.total_length_emb = JukeboxRangeEmbedding(1, timing_dims, total_length_range, embed_dim)
            self.absolute_pos_emb = JukeboxRangeEmbedding(music_tokens_shape, timing_dims, absolute_pos_range, embed_dim)
            self.relative_pos_emb = JukeboxRangeEmbedding(music_tokens_shape, timing_dims, relative_pos_range, embed_dim, clamp=True)

    def forward(self, metadata):
        total_length = metadata[:, 0:1]
        offset = metadata[:, 1:2]
        length = metadata[:, 2:3]
        artist = metadata[:, 3:4]
        genre = metadata[:, 4:]
        artist_emb = self.artist_emb(artist)
        mask = (genre >= 0).float().unsqueeze(2)
        genre_emb = (self.bow_genre_emb(genre.clamp(0)) * mask).sum(dim=1, keepdim=True)
        start_emb = genre_emb + artist_emb
        if self.include_time_signal:
            start, end = (offset, offset + length)
            total_length = total_length.float()
            start = start.float()
            end = end.float()
            pos_emb = self.total_length_emb(total_length) + self.absolute_pos_emb(start, end) + self.relative_pos_emb(start / total_length, end / total_length)
        else:
            pos_emb = None
        return (start_emb, pos_emb)