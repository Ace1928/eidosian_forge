import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class UnclosedFLACFileStream(pyogg.FlacFileStream):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.seekable = True

    def __del__(self):
        if self.decoder:
            pyogg.flac.FLAC__stream_decoder_finish(self.decoder)