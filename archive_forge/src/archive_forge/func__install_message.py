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
def _install_message(self, message):
    """Write message contents and return (start, stop)."""
    start = self._file.tell()
    if isinstance(message, BabylMessage):
        special_labels = []
        labels = []
        for label in message.get_labels():
            if label in self._special_labels:
                special_labels.append(label)
            else:
                labels.append(label)
        self._file.write(b'1')
        for label in special_labels:
            self._file.write(b', ' + label.encode())
        self._file.write(b',,')
        for label in labels:
            self._file.write(b' ' + label.encode() + b',')
        self._file.write(linesep)
    else:
        self._file.write(b'1,,' + linesep)
    if isinstance(message, email.message.Message):
        orig_buffer = io.BytesIO()
        orig_generator = email.generator.BytesGenerator(orig_buffer, False, 0)
        orig_generator.flatten(message)
        orig_buffer.seek(0)
        while True:
            line = orig_buffer.readline()
            self._file.write(line.replace(b'\n', linesep))
            if line == b'\n' or not line:
                break
        self._file.write(b'*** EOOH ***' + linesep)
        if isinstance(message, BabylMessage):
            vis_buffer = io.BytesIO()
            vis_generator = email.generator.BytesGenerator(vis_buffer, False, 0)
            vis_generator.flatten(message.get_visible())
            while True:
                line = vis_buffer.readline()
                self._file.write(line.replace(b'\n', linesep))
                if line == b'\n' or not line:
                    break
        else:
            orig_buffer.seek(0)
            while True:
                line = orig_buffer.readline()
                self._file.write(line.replace(b'\n', linesep))
                if line == b'\n' or not line:
                    break
        while True:
            buffer = orig_buffer.read(4096)
            if not buffer:
                break
            self._file.write(buffer.replace(b'\n', linesep))
    elif isinstance(message, (bytes, str, io.StringIO)):
        if isinstance(message, io.StringIO):
            warnings.warn('Use of StringIO input is deprecated, use BytesIO instead', DeprecationWarning, 3)
            message = message.getvalue()
        if isinstance(message, str):
            message = self._string_to_bytes(message)
        body_start = message.find(b'\n\n') + 2
        if body_start - 2 != -1:
            self._file.write(message[:body_start].replace(b'\n', linesep))
            self._file.write(b'*** EOOH ***' + linesep)
            self._file.write(message[:body_start].replace(b'\n', linesep))
            self._file.write(message[body_start:].replace(b'\n', linesep))
        else:
            self._file.write(b'*** EOOH ***' + linesep + linesep)
            self._file.write(message.replace(b'\n', linesep))
    elif hasattr(message, 'readline'):
        if hasattr(message, 'buffer'):
            warnings.warn('Use of text mode files is deprecated, use a binary mode file instead', DeprecationWarning, 3)
            message = message.buffer
        original_pos = message.tell()
        first_pass = True
        while True:
            line = message.readline()
            if line.endswith(b'\r\n'):
                line = line[:-2] + b'\n'
            elif line.endswith(b'\r'):
                line = line[:-1] + b'\n'
            self._file.write(line.replace(b'\n', linesep))
            if line == b'\n' or not line:
                if first_pass:
                    first_pass = False
                    self._file.write(b'*** EOOH ***' + linesep)
                    message.seek(original_pos)
                else:
                    break
        while True:
            line = message.readline()
            if not line:
                break
            if line.endswith(b'\r\n'):
                line = line[:-2] + linesep
            elif line.endswith(b'\r'):
                line = line[:-1] + linesep
            elif line.endswith(b'\n'):
                line = line[:-1] + linesep
            self._file.write(line)
    else:
        raise TypeError('Invalid message type: %s' % type(message))
    stop = self._file.tell()
    return (start, stop)