from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def _complete_dispatcher(self, dt):
    """This method is scheduled on all touch up events. It will dispatch
        the `on_gesture_complete` event for all completed gestures, and remove
        merged gestures from the internal gesture list."""
    need_cleanup = False
    gest = self._gestures
    timeout = self.draw_timeout
    twin = self.temporal_window
    get_time = Clock.get_time
    for idx, g in enumerate(gest):
        if g.was_merged:
            del gest[idx]
            continue
        if not g.active or g.active_strokes != 0:
            continue
        t1 = g._update_time + twin
        t2 = get_time() + UNDERSHOOT_MARGIN
        if not g.accept_stroke() or t1 <= t2:
            discard = False
            if g.width < 5 and g.height < 5:
                discard = True
            elif g.single_points_test():
                discard = True
            need_cleanup = True
            g.active = False
            g._cleanup_time = get_time() + timeout
            if discard:
                self.dispatch('on_gesture_discard', g)
            else:
                self.dispatch('on_gesture_complete', g)
    if need_cleanup:
        Clock.schedule_once(self._cleanup, timeout)