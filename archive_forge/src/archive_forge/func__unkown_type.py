import gi
from gi.repository import GLib, Gst
import sys
import threading
import os
import queue
from urllib.parse import quote
from .exceptions import DecodeError
from .base import AudioFile
def _unkown_type(self, uridecodebin, decodebin, caps):
    """The callback for decodebin's "unknown-type" signal.
        """
    streaminfo = caps.to_string()
    if not streaminfo.startswith('audio/'):
        return
    self.read_exc = UnknownTypeError(streaminfo)
    self.ready_sem.release()