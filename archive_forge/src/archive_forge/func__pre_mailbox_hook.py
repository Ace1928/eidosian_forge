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
def _pre_mailbox_hook(self, f):
    """Called before writing the mailbox to file f."""
    babyl = b'BABYL OPTIONS:' + linesep
    babyl += b'Version: 5' + linesep
    labels = self.get_labels()
    labels = (label.encode() for label in labels)
    babyl += b'Labels:' + b','.join(labels) + linesep
    babyl += b'\x1f'
    f.write(babyl)