import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
def _create_carefully(path):
    """Create a file if it doesn't exist and open for reading and writing."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 438)
    try:
        return open(path, 'rb+')
    finally:
        os.close(fd)