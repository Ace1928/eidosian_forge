from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@dataclass
class SourceAudioStream(SourceStream):
    """The metadata of an audio source stream, returned by :meth:`~torio.io.StreamingMediaDecoder.get_src_stream_info`.

    This class is used when representing audio stream.

    In addition to the attributes reported by :class:`SourceStream`,
    the following attributes are reported.
    """
    sample_rate: float
    'Sample rate of the audio.'
    num_channels: int
    'Number of channels.'