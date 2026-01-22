from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class NoSysPath(PathError):
    """The filesystem does not provide *sys paths* to the resource."""
    default_message = "path '{path}' does not map to the local filesystem"