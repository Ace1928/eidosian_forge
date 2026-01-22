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
class SmartClientMediumRequest:
    """A request on a SmartClientMedium.

    Each request allows bytes to be provided to it via accept_bytes, and then
    the response bytes to be read via read_bytes.

    For instance:
    request.accept_bytes('123')
    request.finished_writing()
    result = request.read_bytes(3)
    request.finished_reading()

    It is up to the individual SmartClientMedium whether multiple concurrent
    requests can exist. See SmartClientMedium.get_request to obtain instances
    of SmartClientMediumRequest, and the concrete Medium you are using for
    details on concurrency and pipelining.
    """

    def __init__(self, medium):
        """Construct a SmartClientMediumRequest for the medium medium."""
        self._medium = medium
        self._state = 'writing'

    def accept_bytes(self, bytes):
        """Accept bytes for inclusion in this request.

        This method may not be called after finished_writing() has been
        called.  It depends upon the Medium whether or not the bytes will be
        immediately transmitted. Message based Mediums will tend to buffer the
        bytes until finished_writing() is called.

        :param bytes: A bytestring.
        """
        if self._state != 'writing':
            raise errors.WritingCompleted(self)
        self._accept_bytes(bytes)

    def _accept_bytes(self, bytes):
        """Helper for accept_bytes.

        Accept_bytes checks the state of the request to determing if bytes
        should be accepted. After that it hands off to _accept_bytes to do the
        actual acceptance.
        """
        raise NotImplementedError(self._accept_bytes)

    def finished_reading(self):
        """Inform the request that all desired data has been read.

        This will remove the request from the pipeline for its medium (if the
        medium supports pipelining) and any further calls to methods on the
        request will raise ReadingCompleted.
        """
        if self._state == 'writing':
            raise errors.WritingNotComplete(self)
        if self._state != 'reading':
            raise errors.ReadingCompleted(self)
        self._state = 'done'
        self._finished_reading()

    def _finished_reading(self):
        """Helper for finished_reading.

        finished_reading checks the state of the request to determine if
        finished_reading is allowed, and if it is hands off to _finished_reading
        to perform the action.
        """
        raise NotImplementedError(self._finished_reading)

    def finished_writing(self):
        """Finish the writing phase of this request.

        This will flush all pending data for this request along the medium.
        After calling finished_writing, you may not call accept_bytes anymore.
        """
        if self._state != 'writing':
            raise errors.WritingCompleted(self)
        self._state = 'reading'
        self._finished_writing()

    def _finished_writing(self):
        """Helper for finished_writing.

        finished_writing checks the state of the request to determine if
        finished_writing is allowed, and if it is hands off to _finished_writing
        to perform the action.
        """
        raise NotImplementedError(self._finished_writing)

    def read_bytes(self, count):
        """Read bytes from this requests response.

        This method will block and wait for count bytes to be read. It may not
        be invoked until finished_writing() has been called - this is to ensure
        a message-based approach to requests, for compatibility with message
        based mediums like HTTP.
        """
        if self._state == 'writing':
            raise errors.WritingNotComplete(self)
        if self._state != 'reading':
            raise errors.ReadingCompleted(self)
        return self._read_bytes(count)

    def _read_bytes(self, count):
        """Helper for SmartClientMediumRequest.read_bytes.

        read_bytes checks the state of the request to determing if bytes
        should be read. After that it hands off to _read_bytes to do the
        actual read.

        By default this forwards to self._medium.read_bytes because we are
        operating on the medium's stream.
        """
        return self._medium.read_bytes(count)

    def read_line(self):
        line = self._read_line()
        if not line.endswith(b'\n'):
            raise errors.ConnectionReset('Unexpected end of message. Please check connectivity and permissions, and report a bug if problems persist.')
        return line

    def _read_line(self):
        """Helper for SmartClientMediumRequest.read_line.

        By default this forwards to self._medium._get_line because we are
        operating on the medium's stream.
        """
        return self._medium._get_line()