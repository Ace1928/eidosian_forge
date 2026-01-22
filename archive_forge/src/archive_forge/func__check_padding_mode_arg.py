from __future__ import annotations
import collections.abc
import numbers
from contextlib import suppress
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision._utils import sequence_to_str
from torchvision.transforms.transforms import _check_sequence_input, _setup_angle, _setup_size  # noqa: F401
from torchvision.transforms.v2.functional import get_dimensions, get_size, is_pure_tensor
from torchvision.transforms.v2.functional._utils import _FillType, _FillTypeJIT
def _check_padding_mode_arg(padding_mode: Literal['constant', 'edge', 'reflect', 'symmetric']) -> None:
    if padding_mode not in ['constant', 'edge', 'reflect', 'symmetric']:
        raise ValueError('Padding mode should be either constant, edge, reflect or symmetric')