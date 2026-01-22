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
class FileSource(object):

    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.kwargs = kwargs

    def open(self, mode='r'):
        return io.open(self.filename, mode, **self.kwargs)