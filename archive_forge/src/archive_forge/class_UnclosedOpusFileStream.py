import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class UnclosedOpusFileStream(pyogg.OpusFileStream):

    def __del__(self):
        self.ptr.contents.value = self.ptr_init
        del self.ptr
        if self.of:
            pyogg.opus.op_free(self.of)

    def clean_up(self):
        pass