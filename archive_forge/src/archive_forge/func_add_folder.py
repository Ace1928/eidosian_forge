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
def add_folder(self, folder):
    """Create a folder and return an MH instance representing it."""
    return MH(os.path.join(self._path, folder), factory=self._factory)