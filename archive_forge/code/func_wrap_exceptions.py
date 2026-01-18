import errno
import io
import os
import secrets
import shutil
from contextlib import suppress
from functools import cached_property, wraps
from urllib.parse import parse_qs
from fsspec.spec import AbstractFileSystem
from fsspec.utils import (
def wrap_exceptions(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as exception:
            if not exception.args:
                raise
            message, *args = exception.args
            if isinstance(message, str) and 'does not exist' in message:
                raise FileNotFoundError(errno.ENOENT, message) from exception
            else:
                raise
    return wrapper