import io
from typing import Iterator, List, Optional
import torch
from torch import Tensor
from torio.io._streaming_media_decoder import _get_afilter_desc, StreamingMediaDecoder as StreamReader
from torio.io._streaming_media_encoder import CodecConfig, StreamingMediaEncoder as StreamWriter
Apply the effect and/or codecs to the given tensor chunk by chunk.

        Args:
            waveform (Tensor): The input waveform. Shape: ``(time, channel)``
            sample_rate (int): Sample rate of the waveform.
            frames_per_chunk (int): The number of frames to return at a time.
            output_sample_rate (int or None, optional): Output sample rate.
                If provided, override the output sample rate.
                Otherwise, the resulting tensor is resampled to have
                the same sample rate as the input.
                Default: ``None``.

        Returns:
            Iterator[Tensor]:
                Series of processed chunks. Shape: ``(time, channel)``, where the
                the number of frames matches ``frames_per_chunk`` except the
                last chunk, which could be shorter.
        