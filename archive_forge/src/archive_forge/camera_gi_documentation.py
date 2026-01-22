from gi.repository import Gst
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.camera import CameraBase
from kivy.support import install_gobject_iteration
from kivy.logger import Logger
from ctypes import Structure, c_void_p, c_int, string_at
from weakref import ref
import atexit
Implementation of CameraBase using GStreamer

    :Parameters:
        `video_src`: str, default is 'v4l2src'
            Other tested options are: 'dc1394src' for firewire
            dc camera (e.g. firefly MV). Any gstreamer video source
            should potentially work.
            Theoretically a longer string using "!" can be used
            describing the first part of a gstreamer pipeline.
    