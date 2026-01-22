from weakref import ref
from time import time
from kivy.core.text import DEFAULT_FONT
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.utils import platform as core_platform
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import (
import collections.abc
from os import listdir
from os.path import (
from fnmatch import fnmatch
class FileChooserProgressBase(FloatLayout):
    """Base for implementing a progress view. This view is used when too many
    entries need to be created and are delayed over multiple frames.

    .. versionadded:: 1.2.0
    """
    path = StringProperty('')
    'Current path of the FileChooser, read-only.\n    '
    index = NumericProperty(0)
    'Current index of :attr:`total` entries to be loaded.\n    '
    total = NumericProperty(1)
    'Total number of entries to load.\n    '

    def cancel(self, *largs):
        """Cancel any action from the FileChooserController.
        """
        if self.parent:
            self.parent.cancel()

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            super(FileChooserProgressBase, self).on_touch_down(touch)
            return True

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            super(FileChooserProgressBase, self).on_touch_move(touch)
            return True

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            super(FileChooserProgressBase, self).on_touch_up(touch)
            return True