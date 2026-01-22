from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components
def _get_conformer_negativer_sampler(input_dim: int, output_dim: int, num_negatives: int, cross_sample_negatives: int) -> NegativeSampler:
    """Build custom NegativeSampler module, including linear layer and negative sampling.

    Args:
        input_dim (int): Dimension of input after feature extraction.
        output_dim (int): Dimension of embedding for use in negative sampling. Same as the
            embedding in the feature projection.
        num_negatives (int): Number of negatives to sample.
        cross_sample_negatives (int): Number of cross sampled negatives.

    Returns:
        NegativeSampler:
            The resulting negative sampler module.
    """
    preprocessor = nn.Linear(input_dim, output_dim)
    return NegativeSampler(preprocessor, num_negatives, cross_sample_negatives)