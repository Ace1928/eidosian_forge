import sys
from collections import deque
from ctypes import (c_int, c_int32, c_uint8, c_char_p,
import pyglet
import pyglet.lib
from pyglet import image
from pyglet.util import asbytes, asstr
from . import MediaDecoder
from .base import AudioData, SourceInfo, StaticSource
from .base import StreamingSource, VideoFormat, AudioFormat
from .ffmpeg_lib import *
from ..exceptions import MediaFormatException
def _decode_audio_packet(self, audio_packet, compensation_time):
    while True:
        try:
            size_out = self._ffmpeg_decode_audio(audio_packet.packet, self._audio_buffer, compensation_time)
        except FFmpegException:
            break
        if size_out <= 0:
            break
        buffer = create_string_buffer(size_out)
        memmove(buffer, self._audio_buffer, len(buffer))
        buffer = buffer.raw
        duration = float(len(buffer)) / self.audio_format.bytes_per_second
        timestamp = ffmpeg_get_frame_ts(self._audio_stream)
        timestamp = timestamp_from_ffmpeg(timestamp)
        return (buffer, timestamp, duration)
    return (None, 0, 0)