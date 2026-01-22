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
def _decode_video_packet(self, video_packet):
    width = self.video_format.width
    height = self.video_format.height
    pitch = width * 4
    nbytes = pitch * height + FF_INPUT_BUFFER_PADDING_SIZE
    buffer = (c_uint8 * nbytes)()
    try:
        result = self._ffmpeg_decode_video(video_packet.packet, buffer)
    except FFmpegException:
        image_data = None
    else:
        image_data = image.ImageData(width, height, 'RGBA', buffer, pitch)
        timestamp = ffmpeg_get_frame_ts(self._video_stream)
        timestamp = timestamp_from_ffmpeg(timestamp)
        video_packet.timestamp = timestamp - self.start_time
    video_packet.image = image_data
    if _debug:
        print('Decoding video packet at timestamp', video_packet, video_packet.timestamp)