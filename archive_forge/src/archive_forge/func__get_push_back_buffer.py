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
def _get_push_back_buffer(self):
    if self._push_back_buffer == b'':
        raise AssertionError('%s._push_back_buffer should never be the empty string, which can be confused with EOF' % (self,))
    bytes = self._push_back_buffer
    self._push_back_buffer = None
    return bytes