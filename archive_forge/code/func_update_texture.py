from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
def update_texture(self, dt: float=None) -> None:
    """Manually update the texture from the current source.

        This happens automatically, so you shouldn't need to call this method.

        Args:
            dt (float): The time elapsed since the last call to
                ``update_texture``.
        """
    source = self.source
    time = self.time
    if bl.logger is not None:
        bl.logger.log('p.P.ut.1.0', dt, time, self._audio_player.get_time() if self._audio_player else 0, bl.logger.rebased_wall_time())
    frame_rate = source.video_format.frame_rate
    frame_duration = 1 / frame_rate
    ts = source.get_next_video_timestamp()
    while ts is not None and ts + frame_duration < time:
        source.get_next_video_frame()
        if bl.logger is not None:
            bl.logger.log('p.P.ut.1.5', ts)
        ts = source.get_next_video_timestamp()
    if bl.logger is not None:
        bl.logger.log('p.P.ut.1.6', ts)
    if ts is None:
        if bl.logger is not None:
            bl.logger.log('p.P.ut.1.7', frame_duration)
        pyglet.clock.schedule_once(self._video_finished, 0)
        return
    elif ts > time:
        pyglet.clock.schedule_once(self.update_texture, ts - time)
        return
    image = source.get_next_video_frame()
    if image is not None:
        with self._context:
            if self._texture is None:
                self._create_texture()
            self._texture.blit_into(image, 0, 0, 0)
    elif bl.logger is not None:
        bl.logger.log('p.P.ut.1.8')
    ts = source.get_next_video_timestamp()
    if ts is None:
        delay = frame_duration
    else:
        delay = ts - time
    delay = max(0.0, delay)
    if bl.logger is not None:
        bl.logger.log('p.P.ut.1.9', delay, ts)
    pyglet.clock.schedule_once(self.update_texture, delay)