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
def _apply_filters(self, files):
    if not self.filters:
        return files
    filtered = []
    for filt in self.filters:
        if isinstance(filt, collections.abc.Callable):
            filtered.extend([fn for fn in files if filt(self.path, fn)])
        else:
            filtered.extend([fn for fn in files if fnmatch(fn, filt)])
    if not self.filter_dirs:
        dirs = [fn for fn in files if self.file_system.is_dir(fn)]
        filtered.extend(dirs)
    return list(set(filtered))