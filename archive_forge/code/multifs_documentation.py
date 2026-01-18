from __future__ import absolute_import, print_function, unicode_literals
import typing
from collections import OrderedDict, namedtuple
from operator import itemgetter
from six import text_type
from . import errors
from .base import FS
from .mode import check_writable
from .opener import open_fs
from .path import abspath, normpath
Get a tuple of (name, fs) that the given path would map to.

        Arguments:
            path (str): A path on the filesystem.
            mode (str): An `io.open` mode.

        