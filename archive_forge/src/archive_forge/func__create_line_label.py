import re
import sys
import math
from os import environ
from weakref import ref
from itertools import chain, islice
from kivy.animation import Animation
from kivy.base import EventLoop
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.metrics import inch
from kivy.utils import boundary, platform
from kivy.uix.behaviors import FocusBehavior
from kivy.core.text import Label, DEFAULT_FONT
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix, Callback
from kivy.graphics.context_instructions import Transform
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.bubble import Bubble
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty, \
def _create_line_label(self, text, hint=False):
    ntext = text.replace(u'\n', u'').replace(u'\t', u' ' * self.tab_width)
    if self.password and (not hint):
        ntext = self.password_mask * len(ntext)
    kw = self._get_line_options()
    cid = '%s\x00%s' % (ntext, str(kw))
    texture = Cache_get('textinput.label', cid)
    if texture is None:
        label = None
        label_len = len(ntext)
        ld = None
        if not ntext:
            texture = Texture.create(size=(1, 1))
            Cache_append('textinput.label', cid, texture)
            return texture
        while True:
            try:
                label = Label(text=ntext[:label_len], **kw)
                label.refresh()
                if ld is not None and ld > 2:
                    ld //= 2
                    label_len += ld
                else:
                    break
            except:
                if ld is None:
                    ld = len(ntext)
                ld //= 2
                if ld < 2 and label_len:
                    label_len -= 1
                label_len -= ld
                continue
        texture = label.texture
        Cache_append('textinput.label', cid, texture)
    return texture