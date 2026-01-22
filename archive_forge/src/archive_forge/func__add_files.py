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
def _add_files(self, path, parent=None):
    path = expanduser(path)
    if isfile(path):
        path = dirname(path)
    files = []
    fappend = files.append
    for f in self.file_system.listdir(path):
        try:
            fappend(normpath(join(path, f)))
        except UnicodeDecodeError:
            Logger.exception('unable to decode <{}>'.format(f))
        except UnicodeEncodeError:
            Logger.exception('unable to encode <{}>'.format(f))
    files = self._apply_filters(files)
    files = self.sort_func(files, self.file_system)
    is_hidden = self.file_system.is_hidden
    if not self.show_hidden:
        files = [x for x in files if not is_hidden(x)]
    self.files[:] = files
    total = len(files)
    wself = ref(self)
    for index, fn in enumerate(files):

        def get_nice_size():
            return self.get_nice_size(fn)
        ctx = {'name': basename(fn), 'get_nice_size': get_nice_size, 'path': fn, 'controller': wself, 'isdir': self.file_system.is_dir(fn), 'parent': parent, 'sep': sep}
        entry = self._create_entry_widget(ctx)
        yield (index, total, entry)