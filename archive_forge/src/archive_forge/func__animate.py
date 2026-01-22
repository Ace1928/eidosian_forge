import sys
import pyglet
from pyglet.gl import *
from pyglet import clock
from pyglet import event
from pyglet import graphics
from pyglet import image
def _animate(self, dt):
    self._frame_index += 1
    if self._frame_index >= len(self._animation.frames):
        self._frame_index = 0
        self.dispatch_event('on_animation_end')
        if self._vertex_list is None:
            return
    frame = self._animation.frames[self._frame_index]
    self._set_texture(frame.image.get_texture())
    if frame.duration is not None:
        duration = frame.duration - (self._next_dt - dt)
        duration = min(max(0, duration), frame.duration)
        clock.schedule_once(self._animate, duration)
        self._next_dt = duration
    else:
        self.dispatch_event('on_animation_end')