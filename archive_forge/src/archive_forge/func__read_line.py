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
def _read_line(self):
    """Helper for SmartClientMediumRequest.read_line.

        By default this forwards to self._medium._get_line because we are
        operating on the medium's stream.
        """
    return self._medium._get_line()