import queue
import atexit
import weakref
import tempfile
from threading import Event, Thread
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
@staticmethod
def unknown_type(uridecodebin, decodebin, caps):
    """unknown-type callback for unreadable files"""
    streaminfo = caps.to_string()
    if not streaminfo.startswith('audio/'):
        return
    raise GStreamerDecodeException(streaminfo)