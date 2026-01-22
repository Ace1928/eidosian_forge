from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
class HorizontalBlurEffect(EffectBase):
    """Blurs the input horizontally, with the width given by
    :attr:`~HorizontalBlurEffect.size`."""
    size = NumericProperty(4.0)
    'The blur width in pixels.\n\n    size is a :class:`~kivy.properties.NumericProperty` and defaults to\n    4.0.\n    '

    def __init__(self, *args, **kwargs):
        super(HorizontalBlurEffect, self).__init__(*args, **kwargs)
        self.do_glsl()

    def on_size(self, *args):
        self.do_glsl()

    def do_glsl(self):
        self.glsl = effect_blur_h.format(float(self.size))