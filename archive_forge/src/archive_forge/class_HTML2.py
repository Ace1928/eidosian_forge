import os
from base64 import b64encode
from moviepy.audio.AudioClip import AudioClip
from moviepy.tools import extensions_dict
from ..VideoClip import ImageClip, VideoClip
from .ffmpeg_reader import ffmpeg_parse_infos
class HTML2(HTML):

    def __add__(self, other):
        return HTML2(self.data + other.data)