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
def _accept_bytes(self, bytes):
    """See SmartClientMediumRequest._accept_bytes.

        This forwards to self._medium._accept_bytes because we are operating
        on the mediums stream.
        """
    self._medium._accept_bytes(bytes)