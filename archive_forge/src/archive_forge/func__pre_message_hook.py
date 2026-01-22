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
def _pre_message_hook(self, f):
    """Called before writing each message to file f."""
    f.write(b'\x0c' + linesep)