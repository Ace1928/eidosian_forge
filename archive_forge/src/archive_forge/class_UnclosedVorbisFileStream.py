import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class UnclosedVorbisFileStream(pyogg.VorbisFileStream):

    def __del__(self):
        if self.exists:
            pyogg.vorbis.ov_clear(byref(self.vf))
        self.exists = False

    def clean_up(self):
        """PyOgg calls clean_up on end of data. We may want to loop a sound or replay. Prevent this.
        Rely on GC (__del__) to clean up objects instead.
        """
        return