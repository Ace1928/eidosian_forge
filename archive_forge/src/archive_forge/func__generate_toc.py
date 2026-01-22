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
def _generate_toc(self):
    """Generate key-to-(start, stop) table of contents."""
    starts, stops = ([], [])
    self._file.seek(0)
    next_pos = 0
    label_lists = []
    while True:
        line_pos = next_pos
        line = self._file.readline()
        next_pos = self._file.tell()
        if line == b'\x1f\x0c' + linesep:
            if len(stops) < len(starts):
                stops.append(line_pos - len(linesep))
            starts.append(next_pos)
            labels = [label.strip() for label in self._file.readline()[1:].split(b',') if label.strip()]
            label_lists.append(labels)
        elif line == b'\x1f' or line == b'\x1f' + linesep:
            if len(stops) < len(starts):
                stops.append(line_pos - len(linesep))
        elif not line:
            stops.append(line_pos - len(linesep))
            break
    self._toc = dict(enumerate(zip(starts, stops)))
    self._labels = dict(enumerate(label_lists))
    self._next_key = len(self._toc)
    self._file.seek(0, 2)
    self._file_length = self._file.tell()