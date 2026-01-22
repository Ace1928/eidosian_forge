from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@dataclass
class SourceStream:
    """The metadata of a source stream, returned by :meth:`~torio.io.StreamingMediaDecoder.get_src_stream_info`.

    This class is used when representing streams of media type other than `audio` or `video`.

    When source stream is `audio` or `video` type, :class:`SourceAudioStream` and
    :class:`SourceVideoStream`, which reports additional media-specific attributes,
    are used respectively.
    """
    media_type: str
    'The type of the stream.\n    One of ``"audio"``, ``"video"``, ``"data"``, ``"subtitle"``, ``"attachment"`` and empty string.\n\n    .. note::\n       Only audio and video streams are supported for output.\n    .. note::\n       Still images, such as PNG and JPEG formats are reported as video.\n    '
    codec: str
    'Short name of the codec. Such as ``"pcm_s16le"`` and ``"h264"``.'
    codec_long_name: str
    'Detailed name of the codec.\n\n    Such as "`PCM signed 16-bit little-endian`" and "`H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10`".\n    '
    format: Optional[str]
    'Media format. Such as ``"s16"`` and ``"yuv420p"``.\n\n    Commonly found audio values are;\n\n    - ``"u8"``, ``"u8p"``: Unsigned 8-bit unsigned interger.\n    - ``"s16"``, ``"s16p"``: 16-bit signed integer.\n    - ``"s32"``, ``"s32p"``: 32-bit signed integer.\n    - ``"flt"``, ``"fltp"``: 32-bit floating-point.\n\n    .. note::\n\n       `p` at the end indicates the format is `planar`.\n       Channels are grouped together instead of interspersed in memory.\n    '
    bit_rate: Optional[int]
    'Bit rate of the stream in bits-per-second.\n    This is an estimated values based on the initial few frames of the stream.\n    For container formats and variable bit rate, it can be 0.\n    '
    num_frames: Optional[int]
    'The number of frames in the stream'
    bits_per_sample: Optional[int]
    'This is the number of valid bits in each output sample.\n    For compressed format, it can be 0.\n    '
    metadata: Dict[str, str]
    'Metadata attached to the source stream.'