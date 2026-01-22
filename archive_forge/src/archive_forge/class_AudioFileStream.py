from __future__ import annotations
import aifc
import audioop
import base64
import collections
import hashlib
import hmac
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from .audio import AudioData, get_flac_converter
from .exceptions import (
class AudioFileStream(object):

    def __init__(self, audio_reader, little_endian, samples_24_bit_pretending_to_be_32_bit):
        self.audio_reader = audio_reader
        self.little_endian = little_endian
        self.samples_24_bit_pretending_to_be_32_bit = samples_24_bit_pretending_to_be_32_bit

    def read(self, size=-1):
        buffer = self.audio_reader.readframes(self.audio_reader.getnframes() if size == -1 else size)
        if not isinstance(buffer, bytes):
            buffer = b''
        sample_width = self.audio_reader.getsampwidth()
        if not self.little_endian:
            if hasattr(audioop, 'byteswap'):
                buffer = audioop.byteswap(buffer, sample_width)
            else:
                buffer = buffer[sample_width - 1::-1] + b''.join((buffer[i + sample_width:i:-1] for i in range(sample_width - 1, len(buffer), sample_width)))
        if self.samples_24_bit_pretending_to_be_32_bit:
            buffer = b''.join((b'\x00' + buffer[i:i + sample_width] for i in range(0, len(buffer), sample_width)))
            sample_width = 4
        if self.audio_reader.getnchannels() != 1:
            buffer = audioop.tomono(buffer, sample_width, 1, 1)
        return buffer