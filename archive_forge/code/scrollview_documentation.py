from functools import partial
from kivy.animation import Animation
from kivy.compat import string_types
from kivy.config import Config
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.uix.stencilview import StencilView
from kivy.metrics import dp
from kivy.effects.dampedscroll import DampedScrollEffect
from kivy.properties import NumericProperty, BooleanProperty, AliasProperty, \
from kivy.uix.behaviors import FocusBehavior
Force the reposition of the content, according to current value of
        :attr:`scroll_x` and :attr:`scroll_y`.

        This method is automatically called when one of the :attr:`scroll_x`,
        :attr:`scroll_y`, :attr:`pos` or :attr:`size` properties change, or
        if the size of the content changes.
        