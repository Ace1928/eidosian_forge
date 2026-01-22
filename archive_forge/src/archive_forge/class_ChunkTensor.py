from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@dataclass
class ChunkTensor(ChunkTensorBase):
    """Decoded media frames with metadata.

    The instance of this class represents the decoded video/audio frames with
    metadata, and the instance itself behave like :py:class:`~torch.Tensor`.

    Client codes can pass instance of this class as-if it's
    :py:class:`~torch.Tensor` class, or call the methods defined on
    :py:class:`~torch.Tensor` class.

    Example:
        >>> # Define input streams
        >>> reader = StreamingMediaDecoder(...)
        >>> reader.add_audio_stream(frames_per_chunk=4000, sample_rate=8000)
        >>> reader.add_video_stream(frames_per_chunk=7, frame_rate=28)
        >>> # Decode the streams and fetch frames
        >>> reader.fill_buffer()
        >>> audio_chunk, video_chunk = reader.pop_chunks()

        >>> # Access metadata
        >>> (audio_chunk.pts, video_chunks.pts)
        (0.0, 0.0)
        >>>
        >>> # The second time the PTS is different
        >>> reader.fill_buffer()
        >>> audio_chunk, video_chunk = reader.pop_chunks()
        >>> (audio_chunk.pts, video_chunks.pts)
        (0.5, 0.25)

        >>> # Call PyTorch ops on chunk
        >>> audio_chunk.shape
        torch.Size([4000, 2]
        >>> power = torch.pow(video_chunk, 2)
        >>>
        >>> # the result is a plain torch.Tensor class
        >>> type(power)
        <class 'torch.Tensor'>
        >>>
        >>> # Metadata is not available on the result
        >>> power.pts
        AttributeError: 'Tensor' object has no attribute 'pts'
    """
    _elem: torch.Tensor
    pts: float
    'Presentation time stamp of the first frame in the chunk.\n\n    Unit: second.\n    '