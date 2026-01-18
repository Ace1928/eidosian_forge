import queue
import atexit
import weakref
import tempfile
from threading import Event, Thread
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
unknown-type callback for unreadable files