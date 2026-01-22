from __future__ import absolute_import, print_function, division
import os
import io
import gzip
import sys
import bz2
import zipfile
from contextlib import contextmanager
import subprocess
import logging
from petl.errors import ArgumentError
from petl.compat import urlopen, StringIO, BytesIO, string_types, PY2
class GzipSource(object):

    def __init__(self, filename, remote=False, **kwargs):
        self.filename = filename
        self.remote = remote
        self.kwargs = kwargs

    @contextmanager
    def open(self, mode='r'):
        if self.remote:
            if not mode.startswith('r'):
                raise ArgumentError('source is read-only')
            filehandle = urlopen(self.filename)
        else:
            filehandle = self.filename
        source = gzip.open(filehandle, mode, **self.kwargs)
        try:
            yield source
        finally:
            source.close()