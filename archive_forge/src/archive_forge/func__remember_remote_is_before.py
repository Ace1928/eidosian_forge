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
def _remember_remote_is_before(self, version_tuple):
    """Tell this medium that the remote side is older the given version.

        :seealso: _is_remote_before
        """
    if self._remote_version_is_before is not None and version_tuple > self._remote_version_is_before:
        trace.mutter('_remember_remote_is_before(%r) called, but _remember_remote_is_before(%r) was called previously.', version_tuple, self._remote_version_is_before)
        if 'hpss' in debug.debug_flags:
            ui.ui_factory.show_warning('_remember_remote_is_before(%r) called, but _remember_remote_is_before(%r) was called previously.' % (version_tuple, self._remote_version_is_before))
        return
    self._remote_version_is_before = version_tuple