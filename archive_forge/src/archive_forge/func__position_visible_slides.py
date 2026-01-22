from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
def _position_visible_slides(self, *args):
    slides, index = (self.slides, self.index)
    no_of_slides = len(slides) - 1
    if not slides:
        return
    x, y, width, height = (self.x, self.y, self.width, self.height)
    _offset, direction = (self._offset, self.direction[0])
    _prev, _next, _current = (self._prev, self._next, self._current)
    get_slide_container = self.get_slide_container
    last_slide = get_slide_container(slides[-1])
    first_slide = get_slide_container(slides[0])
    skip_next = False
    _loop = self.loop
    if direction in 'rl':
        xoff = x + _offset
        x_prev = {'l': xoff + width, 'r': xoff - width}
        x_next = {'l': xoff - width, 'r': xoff + width}
        if _prev:
            _prev.pos = (x_prev[direction], y)
        elif _loop and _next and (index == 0):
            if _offset > 0 and direction == 'r' or (_offset < 0 and direction == 'l'):
                last_slide.pos = (x_prev[direction], y)
                skip_next = True
        if _current:
            _current.pos = (xoff, y)
        if skip_next:
            return
        if _next:
            _next.pos = (x_next[direction], y)
        elif _loop and _prev and (index == no_of_slides):
            if _offset < 0 and direction == 'r' or (_offset > 0 and direction == 'l'):
                first_slide.pos = (x_next[direction], y)
    if direction in 'tb':
        yoff = y + _offset
        y_prev = {'t': yoff - height, 'b': yoff + height}
        y_next = {'t': yoff + height, 'b': yoff - height}
        if _prev:
            _prev.pos = (x, y_prev[direction])
        elif _loop and _next and (index == 0):
            if _offset > 0 and direction == 't' or (_offset < 0 and direction == 'b'):
                last_slide.pos = (x, y_prev[direction])
                skip_next = True
        if _current:
            _current.pos = (x, yoff)
        if skip_next:
            return
        if _next:
            _next.pos = (x, y_next[direction])
        elif _loop and _prev and (index == no_of_slides):
            if _offset < 0 and direction == 't' or (_offset > 0 and direction == 'b'):
                first_slide.pos = (x, y_next[direction])