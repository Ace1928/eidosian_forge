from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@_format_video_args
def add_basic_video_stream(self, frames_per_chunk: int, buffer_chunk_size: int=3, *, stream_index: Optional[int]=None, decoder: Optional[str]=None, decoder_option: Optional[Dict[str, str]]=None, format: Optional[str]='rgb24', frame_rate: Optional[int]=None, width: Optional[int]=None, height: Optional[int]=None, hw_accel: Optional[str]=None):
    """Add output video stream

        Args:
            frames_per_chunk (int): {frames_per_chunk}

            buffer_chunk_size (int, optional): {buffer_chunk_size}

            stream_index (int or None, optional): {stream_index}

            decoder (str or None, optional): {decoder}

            decoder_option (dict or None, optional): {decoder_option}

            format (str, optional): Change the format of image channels. Valid values are,

                - ``"rgb24"``: 8 bits * 3 channels (R, G, B)
                - ``"bgr24"``: 8 bits * 3 channels (B, G, R)
                - ``"yuv420p"``: 8 bits * 3 channels (Y, U, V)
                - ``"gray"``: 8 bits * 1 channels

                Default: ``"rgb24"``.

            frame_rate (int or None, optional): If provided, change the frame rate.

            width (int or None, optional): If provided, change the image width. Unit: Pixel.

            height (int or None, optional): If provided, change the image height. Unit: Pixel.

            hw_accel (str or None, optional): {hw_accel}
        """
    self.add_video_stream(frames_per_chunk, buffer_chunk_size, stream_index=stream_index, decoder=decoder, decoder_option=decoder_option, filter_desc=_get_vfilter_desc(frame_rate, width, height, format), hw_accel=hw_accel)