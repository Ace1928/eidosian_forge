from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
class _PlayerProperty:
    """Descriptor for Player attributes to forward to the AudioPlayer.

    We want the Player to have attributes like volume, pitch, etc. These are
    actually implemented by the AudioPlayer. So this descriptor will forward
    an assignement to one of the attributes to the AudioPlayer. For example
    `player.volume = 0.5` will call `player._audio_player.set_volume(0.5)`.

    The Player class has default values at the class level which are retrieved
    if not found on the instance.
    """

    def __init__(self, attribute, doc=None):
        self.private_name = '_' + attribute
        self.setter_name = 'set_' + attribute
        self.__doc__ = doc or ''

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.private_name in obj.__dict__:
            return obj.__dict__[self.private_name]
        return getattr(objtype, self.private_name)

    def __set__(self, obj, value):
        obj.__dict__[self.private_name] = value
        if obj._audio_player:
            getattr(obj._audio_player, self.setter_name)(value)