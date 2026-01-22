from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
def _start_animation(self, *args, **kwargs):
    new_offset = 0
    direction = kwargs.get('direction', self.direction)[0]
    is_horizontal = direction in 'rl'
    extent = self.width if is_horizontal else self.height
    min_move = kwargs.get('min_move', self.min_move)
    _offset = kwargs.get('offset', self._offset)
    if _offset < min_move * -extent:
        new_offset = -extent
    elif _offset > min_move * extent:
        new_offset = extent
    dur = self.anim_move_duration
    if new_offset == 0:
        dur = self.anim_cancel_duration
    len_slides = len(self.slides)
    index = self.index
    if not self.loop or len_slides == 1:
        is_first = index == 0
        is_last = index == len_slides - 1
        if direction in 'rt':
            towards_prev = new_offset > 0
            towards_next = new_offset < 0
        else:
            towards_prev = new_offset < 0
            towards_next = new_offset > 0
        if is_first and towards_prev or (is_last and towards_next):
            new_offset = 0
    anim = Animation(_offset=new_offset, d=dur, t=self.anim_type)
    anim.cancel_all(self)

    def _cmp(*l):
        if self._skip_slide is not None:
            self.index = self._skip_slide
            self._skip_slide = None
    anim.bind(on_complete=_cmp)
    anim.start(self)