import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class MemoryVorbisObject:

    def __init__(self, file):
        self.file = file

        def read_func_cb(ptr, byte_size, size_to_read, datasource):
            data_size = size_to_read * byte_size
            data = self.file.read(data_size)
            read_size = len(data)
            memmove(ptr, data, read_size)
            return read_size

        def seek_func_cb(datasource, offset, whence):
            pos = self.file.seek(offset, whence)
            return pos

        def close_func_cb(datasource):
            return 0

        def tell_func_cb(datasource):
            return self.file.tell()
        self.read_func = pyogg.vorbis.read_func(read_func_cb)
        self.seek_func = pyogg.vorbis.seek_func(seek_func_cb)
        self.close_func = pyogg.vorbis.close_func(close_func_cb)
        self.tell_func = pyogg.vorbis.tell_func(tell_func_cb)
        self.callbacks = pyogg.vorbis.ov_callbacks(self.read_func, self.seek_func, self.close_func, self.tell_func)