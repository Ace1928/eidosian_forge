import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
def _fade_in(self, waveform_length: int, device: torch.device) -> Tensor:
    fade = torch.linspace(0, 1, self.fade_in_len, device=device)
    ones = torch.ones(waveform_length - self.fade_in_len, device=device)
    if self.fade_shape == 'linear':
        fade = fade
    if self.fade_shape == 'exponential':
        fade = torch.pow(2, fade - 1) * fade
    if self.fade_shape == 'logarithmic':
        fade = torch.log10(0.1 + fade) + 1
    if self.fade_shape == 'quarter_sine':
        fade = torch.sin(fade * math.pi / 2)
    if self.fade_shape == 'half_sine':
        fade = torch.sin(fade * math.pi - math.pi / 2) / 2 + 0.5
    return torch.cat((fade, ones)).clamp_(0, 1)