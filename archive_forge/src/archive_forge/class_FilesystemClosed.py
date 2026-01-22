from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class FilesystemClosed(FSError):
    """Attempt to use a closed filesystem."""
    default_message = 'attempt to use closed filesystem'