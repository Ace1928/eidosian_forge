import gi
from gi.repository import GLib, Gst
import sys
import threading
import os
import queue
from urllib.parse import quote
from .exceptions import DecodeError
from .base import AudioFile
def _pad_added(self, element, pad):
    """The callback for GstElement's "pad-added" signal.
        """
    name = pad.query_caps(None).to_string()
    if name.startswith('audio/x-raw'):
        nextpad = self.conv.get_static_pad('sink')
        if not nextpad.is_linked():
            self._got_a_pad = True
            pad.link(nextpad)