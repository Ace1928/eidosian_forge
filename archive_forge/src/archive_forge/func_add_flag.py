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
def add_flag(self, flag):
    """Set the given flag(s) without changing others."""
    self.set_flags(''.join(set(self.get_flags()) | set(flag)))