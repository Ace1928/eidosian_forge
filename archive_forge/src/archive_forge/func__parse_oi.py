from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def _parse_oi(i):
    media_type = i.media_type
    if media_type == 'audio':
        return OutputAudioStream(source_index=i.source_index, filter_description=i.filter_description, media_type=i.media_type, format=i.format, sample_rate=i.sample_rate, num_channels=i.num_channels)
    if media_type == 'video':
        return OutputVideoStream(source_index=i.source_index, filter_description=i.filter_description, media_type=i.media_type, format=i.format, width=i.width, height=i.height, frame_rate=i.frame_rate)
    raise ValueError(f'Unexpected media_type: {i.media_type}({i})')