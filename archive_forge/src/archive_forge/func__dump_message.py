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
def _dump_message(self, message, target, mangle_from_=False):
    """Dump message contents to target file."""
    if isinstance(message, email.message.Message):
        buffer = io.BytesIO()
        gen = email.generator.BytesGenerator(buffer, mangle_from_, 0)
        gen.flatten(message)
        buffer.seek(0)
        data = buffer.read()
        data = data.replace(b'\n', linesep)
        target.write(data)
        if self._append_newline and (not data.endswith(linesep)):
            target.write(linesep)
    elif isinstance(message, (str, bytes, io.StringIO)):
        if isinstance(message, io.StringIO):
            warnings.warn('Use of StringIO input is deprecated, use BytesIO instead', DeprecationWarning, 3)
            message = message.getvalue()
        if isinstance(message, str):
            message = self._string_to_bytes(message)
        if mangle_from_:
            message = message.replace(b'\nFrom ', b'\n>From ')
        message = message.replace(b'\n', linesep)
        target.write(message)
        if self._append_newline and (not message.endswith(linesep)):
            target.write(linesep)
    elif hasattr(message, 'read'):
        if hasattr(message, 'buffer'):
            warnings.warn('Use of text mode files is deprecated, use a binary mode file instead', DeprecationWarning, 3)
            message = message.buffer
        lastline = None
        while True:
            line = message.readline()
            if line.endswith(b'\r\n'):
                line = line[:-2] + b'\n'
            elif line.endswith(b'\r'):
                line = line[:-1] + b'\n'
            if not line:
                break
            if mangle_from_ and line.startswith(b'From '):
                line = b'>From ' + line[5:]
            line = line.replace(b'\n', linesep)
            target.write(line)
            lastline = line
        if self._append_newline and lastline and (not lastline.endswith(linesep)):
            target.write(linesep)
    else:
        raise TypeError('Invalid message type: %s' % type(message))