import gi
from gi.repository import GLib, Gst
import sys
import threading
import os
import queue
from urllib.parse import quote
from .exceptions import DecodeError
from .base import AudioFile
class MainLoopThread(threading.Thread):
    """A daemon thread encapsulating a Gobject main loop.
    """

    def __init__(self):
        super().__init__()
        self.loop = GLib.MainLoop.new(None, False)
        self.daemon = True

    def run(self):
        self.loop.run()