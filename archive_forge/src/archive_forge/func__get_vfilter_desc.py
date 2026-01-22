from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def _get_vfilter_desc(frame_rate: Optional[float], width: Optional[int], height: Optional[int], fmt: Optional[str]):
    descs = []
    if frame_rate is not None:
        descs.append(f'fps={frame_rate}')
    scales = []
    if width is not None:
        scales.append(f'width={width}')
    if height is not None:
        scales.append(f'height={height}')
    if scales:
        descs.append(f'scale={':'.join(scales)}')
    if fmt is not None:
        descs.append(f'format=pix_fmts={fmt}')
    return ','.join(descs) if descs else None