from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
class XAudio2Listener:

    def __init__(self, driver):
        self.xa2_driver = weakref.proxy(driver)
        self.listener = lib.X3DAUDIO_LISTENER()
        self.listener.OrientFront.x = 0
        self.listener.OrientFront.y = 0
        self.listener.OrientFront.z = 1
        self.listener.OrientTop.x = 0
        self.listener.OrientTop.y = 1
        self.listener.OrientTop.z = 0

    def delete(self):
        self.listener = None

    @property
    def position(self):
        return (self.listener.Position.x, self.listener.Position.y, self.listener.Position.z)

    @position.setter
    def position(self, value):
        x, y, z = value
        self.listener.Position.x = x
        self.listener.Position.y = y
        self.listener.Position.z = z

    @property
    def orientation(self):
        return (self.listener.OrientFront.x, self.listener.OrientFront.y, self.listener.OrientFront.z, self.listener.OrientTop.x, self.listener.OrientTop.y, self.listener.OrientTop.z)

    @orientation.setter
    def orientation(self, orientation):
        front_x, front_y, front_z, top_x, top_y, top_z = orientation
        self.listener.OrientFront.x = front_x
        self.listener.OrientFront.y = front_y
        self.listener.OrientFront.z = front_z
        self.listener.OrientTop.x = top_x
        self.listener.OrientTop.y = top_y
        self.listener.OrientTop.z = top_z