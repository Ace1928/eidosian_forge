from __future__ import absolute_import, print_function, unicode_literals
import typing
from six import text_type
from . import errors
from .base import FS
from .memoryfs import MemoryFS
from .mode import validate_open_mode, validate_openbin_mode
from .path import abspath, forcedir, normpath
Mounts a host FS object on a given path.

        Arguments:
            path (str): A path within the MountFS.
            fs (FS or str): A filesystem (instance or URL) to mount.

        