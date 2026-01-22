from __future__ import print_function, unicode_literals
import typing
import array
import calendar
import datetime
import io
import itertools
import socket
import threading
from collections import OrderedDict
from contextlib import contextmanager
from ftplib import FTP
from typing import cast
from ftplib import error_perm, error_temp
from six import PY2, raise_from, text_type
from . import _ftp_parse as ftp_parse
from . import errors
from .base import FS
from .constants import DEFAULT_CHUNK_SIZE
from .enums import ResourceType, Seek
from .info import Info
from .iotools import line_iterator
from .mode import Mode
from .path import abspath, basename, dirname, normpath, split
from .time import epoch_to_datetime
def _open_ftp(self):
    """Open a new ftp object."""
    _ftp = FTP_TLS() if self.tls else FTP()
    _ftp.set_debuglevel(0)
    with ftp_errors(self):
        _ftp.connect(self.host, self.port, self.timeout)
        _ftp.login(self.user, self.passwd, self.acct)
        try:
            _ftp.prot_p()
        except AttributeError:
            pass
        self._features = {}
        try:
            feat_response = _decode(_ftp.sendcmd('FEAT'), 'latin-1')
        except error_perm:
            self.encoding = 'latin-1'
        else:
            self._features = self._parse_features(feat_response)
            self.encoding = 'utf-8' if 'UTF8' in self._features else 'latin-1'
            if not PY2:
                _ftp.file = _ftp.sock.makefile('r', encoding=self.encoding)
    _ftp.encoding = self.encoding
    self._welcome = _ftp.welcome
    return _ftp