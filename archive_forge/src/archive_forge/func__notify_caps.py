import gi
from gi.repository import GLib, Gst
import sys
import threading
import os
import queue
from urllib.parse import quote
from .exceptions import DecodeError
from .base import AudioFile
def _notify_caps(self, pad, args):
    """The callback for the sinkpad's "notify::caps" signal.
        """
    self.got_caps = True
    info = pad.get_current_caps().get_structure(0)
    self.channels = info.get_int('channels')[1]
    self.samplerate = info.get_int('rate')[1]
    success, length = pad.get_peer().query_duration(Gst.Format.TIME)
    if success:
        self.duration = length / 1000000000
    else:
        self.read_exc = MetadataMissingError('duration not available')
    self.ready_sem.release()