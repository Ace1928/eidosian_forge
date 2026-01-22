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
class StdoutSource(object):

    @contextmanager
    def open(self, mode):
        if mode.startswith('r'):
            raise ArgumentError('source is write-only')
        if 'b' in mode:
            yield Uncloseable(stdout_binary)
        else:
            yield Uncloseable(sys.stdout)