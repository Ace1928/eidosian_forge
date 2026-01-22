import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class MemoryOpusFileStream(UnclosedOpusFileStream):

    def __init__(self, filename, file):
        self.file = file
        self.memory_object = MemoryOpusObject(filename, file)
        self._dummy_fileobj = c_void_p()
        error = c_int()
        self.read_buffer = create_string_buffer(pyogg.PYOGG_STREAM_BUFFER_SIZE)
        self.ptr_buffer = cast(self.read_buffer, POINTER(c_ubyte))
        self.of = pyogg.opus.op_open_callbacks(self._dummy_fileobj, byref(self.memory_object.callbacks), self.ptr_buffer, 0, byref(error))
        if error.value != 0:
            raise DecodeException("file-like object: {} couldn't be processed. Error code : {}".format(filename, error.value))
        self.channels = pyogg.opus.op_channel_count(self.of, -1)
        self.pcm_size = pyogg.opus.op_pcm_total(self.of, -1)
        self.frequency = 48000
        self.bfarr_t = pyogg.opus.opus_int16 * (pyogg.PYOGG_STREAM_BUFFER_SIZE * self.channels * 2)
        self.buffer = cast(pointer(self.bfarr_t()), pyogg.opus.opus_int16_p)
        self.ptr = cast(pointer(self.buffer), POINTER(c_void_p))
        self.ptr_init = self.ptr.contents.value