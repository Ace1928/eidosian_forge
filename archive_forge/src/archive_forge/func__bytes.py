import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
@staticmethod
def _bytes(s, encoding):
    """Converts s to bytes if necessary, using encoding.

        If s is already bytes type, returns it directly.
        """
    if isinstance(s, bytes):
        return s
    if isinstance(s, str):
        return s.encode(encoding)
    raise TypeError('bytes or unicode/string required, not %s' % type(s))