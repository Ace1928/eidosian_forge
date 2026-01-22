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
class TvltAudioPatchEmbeddings(nn.Module):
    """
    This class turns `audio_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        spectrogram_length, frequency_length, patch_size = (config.spectrogram_length, config.frequency_length, config.audio_patch_size)
        num_channels, hidden_size = (config.num_audio_channels, config.hidden_size)
        spectrogram_size = (spectrogram_length, frequency_length)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = spectrogram_size[1] // patch_size[1] * (spectrogram_size[0] // patch_size[0])
        patch_shape = (spectrogram_size[0] // patch_size[0], spectrogram_size[1] // patch_size[1])
        self.spectrogram_size = spectrogram_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = audio_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if height > self.spectrogram_size[0] or width != self.spectrogram_size[1]:
            raise ValueError(f"Input audio size ({height}*{width}) doesn't match model ({self.spectrogram_size[0]}*{self.spectrogram_size[1]}).")
        embeddings = self.projection(audio_values).flatten(2).transpose(1, 2)
        return embeddings