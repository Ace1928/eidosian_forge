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
class FileSystemAbstract(object):
    """Class for implementing a File System view that can be used with the
    :class:`FileChooser <FileChooser>`.

    .. versionadded:: 1.8.0
    """

    def listdir(self, fn):
        """Return the list of files in the directory `fn`
        """
        pass

    def getsize(self, fn):
        """Return the size in bytes of a file
        """
        pass

    def is_hidden(self, fn):
        """Return True if the file is hidden
        """
        pass

    def is_dir(self, fn):
        """Return True if the argument passed to this method is a directory
        """
        pass