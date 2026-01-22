import io
from typing import Iterator, List, Optional
import torch
from torch import Tensor
from torio.io._streaming_media_decoder import _get_afilter_desc, StreamingMediaDecoder as StreamReader
from torio.io._streaming_media_encoder import CodecConfig, StreamingMediaEncoder as StreamWriter
class _StreamingIOBuffer:
    """Streaming Bytes IO buffer. Data are dropped when read."""

    def __init__(self):
        self._buffer: List(bytes) = []

    def write(self, b: bytes):
        if b:
            self._buffer.append(b)
        return len(b)

    def pop(self, n):
        """Pop the oldest byte string. It does not necessary return the requested amount"""
        if not self._buffer:
            return b''
        if len(self._buffer[0]) <= n:
            return self._buffer.pop(0)
        ret = self._buffer[0][:n]
        self._buffer[0] = self._buffer[0][n:]
        return ret