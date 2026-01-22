from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components
def _buffered_arange(max) -> Tensor:
    """Compute arange using a buffered tensor across function calls.
    Produces same result as torch.arange(end=max).

    Args:
        max (int): Ending value for arange.
    """
    if not hasattr(_buffered_arange, 'buf'):
        _buffered_arange.buf = torch.LongTensor()
    if max > _buffered_arange.buf.numel():
        _buffered_arange.buf.resize_(max)
        torch.arange(max, out=_buffered_arange.buf)
    return _buffered_arange.buf[:max]