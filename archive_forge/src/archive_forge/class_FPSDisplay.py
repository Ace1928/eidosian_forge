import sys
from typing import Tuple
import pyglet
import pyglet.window.key
import pyglet.window.mouse
from pyglet import gl
from pyglet.math import Mat4
from pyglet.event import EventDispatcher
from pyglet.window import key, event
from pyglet.graphics import shader
class FPSDisplay:
    """Display of a window's framerate.

    This is a convenience class to aid in profiling and debugging.  Typical
    usage is to create an `FPSDisplay` for each window, and draw the display
    at the end of the windows' :py:meth:`~pyglet.window.Window.on_draw` event handler::

        from pyglet.window import Window, FPSDisplay

        window = Window()
        fps_display = FPSDisplay(window)

        @window.event
        def on_draw():
            # ... perform ordinary window drawing operations ...

            fps_display.draw()

    The style and position of the display can be modified via the :py:func:`~pyglet.text.Label`
    attribute.  Different text can be substituted by overriding the
    `set_fps` method.  The display can be set to update more or less often
    by setting the `update_period` attribute. Note: setting the `update_period`
    to a value smaller than your Window refresh rate will cause inaccurate readings.

    :Ivariables:
        `label` : Label
            The text label displaying the framerate.

    """
    update_period = 0.25

    def __init__(self, window, color=(127, 127, 127, 127), samples=240):
        from time import time
        from statistics import mean
        from collections import deque
        from pyglet.text import Label
        self._time = time
        self._mean = mean
        self._window_flip, window.flip = (window.flip, self._hook_flip)
        self.label = Label('', x=10, y=10, font_size=24, bold=True, color=color)
        self._elapsed = 0.0
        self._last_time = time()
        self._delta_times = deque(maxlen=samples)

    def update(self):
        """Records a new data point at the current time. This method
        is called automatically when the window buffer is flipped.
        """
        t = self._time()
        delta = t - self._last_time
        self._elapsed += delta
        self._delta_times.append(delta)
        self._last_time = t
        if self._elapsed >= self.update_period:
            self._elapsed = 0
            self.label.text = f'{1 / self._mean(self._delta_times):.2f}'

    def draw(self):
        """Draw the label."""
        self.label.draw()

    def _hook_flip(self):
        self.update()
        self._window_flip()