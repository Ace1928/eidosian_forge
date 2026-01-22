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
class SmartClientStreamMediumRequest(SmartClientMediumRequest):
    """A SmartClientMediumRequest that works with an SmartClientStreamMedium."""

    def __init__(self, medium):
        SmartClientMediumRequest.__init__(self, medium)
        if self._medium._current_request is not None:
            raise TooManyConcurrentRequests(self._medium)
        self._medium._current_request = self

    def _accept_bytes(self, bytes):
        """See SmartClientMediumRequest._accept_bytes.

        This forwards to self._medium._accept_bytes because we are operating
        on the mediums stream.
        """
        self._medium._accept_bytes(bytes)

    def _finished_reading(self):
        """See SmartClientMediumRequest._finished_reading.

        This clears the _current_request on self._medium to allow a new
        request to be created.
        """
        if self._medium._current_request is not self:
            raise AssertionError()
        self._medium._current_request = None

    def _finished_writing(self):
        """See SmartClientMediumRequest._finished_writing.

        This invokes self._medium._flush to ensure all bytes are transmitted.
        """
        self._medium._flush()