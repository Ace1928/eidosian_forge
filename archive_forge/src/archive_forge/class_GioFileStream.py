import os
import random
import stat
import time
from io import BytesIO
from urllib.parse import urlparse, urlunparse
from .. import config, debug, errors, osutils, ui, urlutils
from ..tests.test_server import TestServer
from ..trace import mutter
from . import (ConnectedTransport, FileExists, FileStream, NoSuchFile,
class GioFileStream(FileStream):
    """A file stream object returned by open_write_stream.

    This version uses GIO to perform writes.
    """

    def __init__(self, transport, relpath):
        FileStream.__init__(self, transport, relpath)
        self.gio_file = transport._get_GIO(relpath)
        self.stream = self.gio_file.create()

    def _close(self):
        self.stream.close()

    def write(self, bytes):
        try:
            osutils.pumpfile(BytesIO(bytes), self.stream)
        except gio.Error as e:
            raise errors.BzrError(str(e))