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
def _get_GIO(self, relpath):
    """Return the ftplib.GIO instance for this object."""
    connection = self._get_connection()
    if connection is None:
        connection, credentials = self._create_connection()
        self._set_connection(connection, credentials)
    fileurl = self._relpath_to_url(relpath)
    file = gio.File(fileurl)
    return file