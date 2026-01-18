from kivy.graphics.texture import Texture
from kivy.core.video import VideoBase
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.compat import PY2
from threading import Lock
from functools import partial
from os.path import realpath
from weakref import ref

Video Gstplayer
===============

.. versionadded:: 1.8.0

Implementation of a VideoBase with Kivy :class:`~kivy.lib.gstplayer.GstPlayer`
This player is the preferred player, using Gstreamer 1.0, working on both
Python 2 and 3.
