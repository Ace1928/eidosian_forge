import gi
from gi.repository import GLib, Gst
import sys
import threading
import os
import queue
from urllib.parse import quote
from .exceptions import DecodeError
from .base import AudioFile
Close the file and clean up associated resources.

        Calling `close()` a second time has no effect.
        