import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
class TextUIOutputStream:
    """Decorates stream to interact better with progress and change encoding.

    Before writing to the wrapped stream, progress is cleared. Callers must
    ensure bulk output is terminated with a newline so progress won't overwrite
    partial lines.

    Additionally, the encoding and errors behaviour of the underlying stream
    can be changed at this point. If errors is set to 'exact' raw bytes may be
    written to the underlying stream.
    """

    def __init__(self, ui_factory, stream, encoding=None, errors='strict'):
        self.ui_factory = ui_factory
        inner = _unwrap_stream(stream)
        self.raw_stream = None
        if errors == 'exact':
            errors = 'strict'
            self.raw_stream = inner
        if inner is None:
            self.wrapped_stream = stream
            if encoding is None:
                encoding = _get_stream_encoding(stream)
        else:
            self.wrapped_stream = _wrap_out_stream(inner, encoding, errors)
            if encoding is None:
                encoding = self.wrapped_stream.encoding
        self.encoding = encoding
        self.errors = errors

    def _write(self, to_write):
        if isinstance(to_write, bytes):
            try:
                to_write = to_write.decode(self.encoding, self.errors)
            except UnicodeDecodeError:
                self.raw_stream.write(to_write)
                return
        self.wrapped_stream.write(to_write)

    def flush(self):
        self.ui_factory.clear_term()
        self.wrapped_stream.flush()

    def write(self, to_write):
        self.ui_factory.clear_term()
        self._write(to_write)

    def writelines(self, lines):
        self.ui_factory.clear_term()
        for line in lines:
            self._write(line)