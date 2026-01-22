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
def _setup_number_or_seq(arg: Union[int, float, Sequence[Union[int, float]]], name: str) -> Sequence[float]:
    if not isinstance(arg, (int, float, Sequence)):
        raise TypeError(f'{name} should be a number or a sequence of numbers. Got {type(arg)}')
    if isinstance(arg, Sequence) and len(arg) not in (1, 2):
        raise ValueError(f'If {name} is a sequence its length should be 1 or 2. Got {len(arg)}')
    if isinstance(arg, Sequence):
        for element in arg:
            if not isinstance(element, (int, float)):
                raise ValueError(f'{name} should be a sequence of numbers. Got {type(element)}')
    if isinstance(arg, (int, float)):
        arg = [float(arg), float(arg)]
    elif isinstance(arg, Sequence):
        if len(arg) == 1:
            arg = [float(arg[0]), float(arg[0])]
        else:
            arg = [float(arg[0]), float(arg[1])]
    return arg