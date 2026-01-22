from kivy.compat import iteritems
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (StringProperty, ObjectProperty, AliasProperty,
from kivy.animation import Animation, AnimationTransition
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
from kivy.graphics import (RenderContext, Rectangle, Fbo,
class CardTransition(SlideTransition):
    """Card transition that looks similar to Android 4.x application drawer
    interface animation.

    It supports 4 directions like SlideTransition: left, right, up and down,
    and two modes, pop and push. If push mode is activated, the previous
    screen does not move, and the new one slides in from the given direction.
    If the pop mode is activated, the previous screen slides out, when the new
    screen is already on the position of the ScreenManager.

    .. versionadded:: 1.10
    """
    mode = OptionProperty('push', options=['pop', 'push'])
    "Indicates if the transition should push or pop\n    the screen on/off the ScreenManager.\n\n    - 'push' means the screen slides in in the given direction\n    - 'pop' means the screen slides out in the given direction\n\n    :attr:`mode` is an :class:`~kivy.properties.OptionProperty` and\n    defaults to 'push'.\n    "

    def start(self, manager):
        """(internal) Starts the transition. This is automatically
        called by the :class:`ScreenManager`.
        """
        super(CardTransition, self).start(manager)
        mode = self.mode
        a = self.screen_in
        b = self.screen_out
        if mode == 'push':
            manager.canvas.remove(a.canvas)
            manager.canvas.add(a.canvas)
        elif mode == 'pop':
            manager.canvas.remove(b.canvas)
            manager.canvas.add(b.canvas)

    def on_progress(self, progression):
        a = self.screen_in
        b = self.screen_out
        manager = self.manager
        x, y = manager.pos
        width, height = manager.size
        direction = self.direction
        mode = self.mode
        al = AnimationTransition.out_quad
        progression = al(progression)
        if mode == 'push':
            b.pos = (x, y)
            if direction == 'left':
                a.pos = (x + width * (1 - progression), y)
            elif direction == 'right':
                a.pos = (x - width * (1 - progression), y)
            elif direction == 'down':
                a.pos = (x, y + height * (1 - progression))
            elif direction == 'up':
                a.pos = (x, y - height * (1 - progression))
        elif mode == 'pop':
            a.pos = (x, y)
            if direction == 'left':
                b.pos = (x - width * progression, y)
            elif direction == 'right':
                b.pos = (x + width * progression, y)
            elif direction == 'down':
                b.pos = (x, y - height * progression)
            elif direction == 'up':
                b.pos = (x, y + height * progression)