from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
def _create_audio_player(self) -> None:
    assert not self._audio_player
    assert self.source
    source = self.source
    audio_driver = get_audio_driver()
    if audio_driver is None:
        return
    self._audio_player = audio_driver.create_audio_player(source, self)
    for attr in ('volume', 'min_distance', 'max_distance', 'position', 'pitch', 'cone_orientation', 'cone_inner_angle', 'cone_outer_angle', 'cone_outer_gain'):
        value = getattr(self, attr)
        setattr(self, attr, value)