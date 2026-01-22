import _thread
import errno
import io
import os
import sys
import time
import breezy
from ...lazy_import import lazy_import
import select
import socket
import weakref
from breezy import (
from breezy.i18n import gettext
from breezy.bzr.smart import client, protocol, request, signals, vfs
from breezy.transport import ssh
from ... import errors, osutils
class SmartSimplePipesClientMedium(SmartClientStreamMedium):
    """A client medium using simple pipes.

    This client does not manage the pipes: it assumes they will always be open.
    """

    def __init__(self, readable_pipe, writeable_pipe, base):
        SmartClientStreamMedium.__init__(self, base)
        self._readable_pipe = readable_pipe
        self._writeable_pipe = writeable_pipe

    def _accept_bytes(self, data):
        """See SmartClientStreamMedium.accept_bytes."""
        try:
            self._writeable_pipe.write(data)
        except OSError as e:
            if e.errno in (errno.EINVAL, errno.EPIPE):
                raise errors.ConnectionReset('Error trying to write to subprocess', e)
            raise
        self._report_activity(len(data), 'write')

    def _flush(self):
        """See SmartClientStreamMedium._flush()."""
        self._writeable_pipe.flush()

    def _read_bytes(self, count):
        """See SmartClientStreamMedium._read_bytes."""
        bytes_to_read = min(count, _MAX_READ_SIZE)
        data = self._readable_pipe.read(bytes_to_read)
        self._report_activity(len(data), 'read')
        return data