import gc
import math
import os
import re
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import _log_api_usage_once
from . import _video_opt
def write_video(filename: str, video_array: torch.Tensor, fps: float, video_codec: str='libx264', options: Optional[Dict[str, Any]]=None, audio_array: Optional[torch.Tensor]=None, audio_fps: Optional[float]=None, audio_codec: Optional[str]=None, audio_options: Optional[Dict[str, Any]]=None) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(write_video)
    _check_av_available()
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()
    if isinstance(fps, float):
        fps = np.round(fps)
    with av.open(filename, mode='w') as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = 'yuv420p' if video_codec != 'libx264rgb' else 'rgb24'
        stream.options = options or {}
        if audio_array is not None:
            audio_format_dtypes = {'dbl': '<f8', 'dblp': '<f8', 'flt': '<f4', 'fltp': '<f4', 's16': '<i2', 's16p': '<i2', 's32': '<i4', 's32p': '<i4', 'u8': 'u1', 'u8p': 'u1'}
            a_stream = container.add_stream(audio_codec, rate=audio_fps)
            a_stream.options = audio_options or {}
            num_channels = audio_array.shape[0]
            audio_layout = 'stereo' if num_channels > 1 else 'mono'
            audio_sample_fmt = container.streams.audio[0].format.name
            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy().astype(format_dtype)
            frame = av.AudioFrame.from_ndarray(audio_array, format=audio_sample_fmt, layout=audio_layout)
            frame.sample_rate = audio_fps
            for packet in a_stream.encode(frame):
                container.mux(packet)
            for packet in a_stream.encode():
                container.mux(packet)
        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            frame.pict_type = 'NONE'
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)