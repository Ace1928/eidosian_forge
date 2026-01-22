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
class SmartServerStreamMedium(SmartMedium):
    """Handles smart commands coming over a stream.

    The stream may be a pipe connected to sshd, or a tcp socket, or an
    in-process fifo for testing.

    One instance is created for each connected client; it can serve multiple
    requests in the lifetime of the connection.

    The server passes requests through to an underlying backing transport,
    which will typically be a LocalTransport looking at the server's filesystem.

    :ivar _push_back_buffer: a str of bytes that have been read from the stream
        but not used yet, or None if there are no buffered bytes.  Subclasses
        should make sure to exhaust this buffer before reading more bytes from
        the stream.  See also the _push_back method.
    """
    _timer = time.time

    def __init__(self, backing_transport, root_client_path='/', timeout=None):
        """Construct new server.

        :param backing_transport: Transport for the directory served.
        """
        self.backing_transport = backing_transport
        self.root_client_path = root_client_path
        self.finished = False
        if timeout is None:
            raise AssertionError('You must supply a timeout.')
        self._client_timeout = timeout
        self._client_poll_timeout = min(timeout / 10.0, 1.0)
        SmartMedium.__init__(self)

    def serve(self):
        """Serve requests until the client disconnects."""
        from sys import stderr
        try:
            while not self.finished:
                server_protocol = self._build_protocol()
                self._serve_one_request(server_protocol)
        except errors.ConnectionTimeout as e:
            trace.note('{}'.format(e))
            trace.log_exception_quietly()
            self._disconnect_client()
            return
        except Exception as e:
            stderr.write('{} terminating on exception {}\n'.format(self, e))
            raise
        self._disconnect_client()

    def _stop_gracefully(self):
        """When we finish this message, stop looking for more."""
        trace.mutter('Stopping {}'.format(self))
        self.finished = True

    def _disconnect_client(self):
        """Close the current connection. We stopped due to a timeout/etc."""

    def _wait_for_bytes_with_timeout(self, timeout_seconds):
        """Wait for more bytes to be read, but timeout if none available.

        This allows us to detect idle connections, and stop trying to read from
        them, without setting the socket itself to non-blocking. This also
        allows us to specify when we watch for idle timeouts.

        :return: Did we timeout? (True if we timed out, False if there is data
            to be read)
        """
        raise NotImplementedError(self._wait_for_bytes_with_timeout)

    def _build_protocol(self):
        """Identifies the version of the incoming request, and returns an
        a protocol object that can interpret it.

        If more bytes than the version prefix of the request are read, they will
        be fed into the protocol before it is returned.

        :returns: a SmartServerRequestProtocol.
        """
        self._wait_for_bytes_with_timeout(self._client_timeout)
        if self.finished:
            return None
        bytes = self._get_line()
        protocol_factory, unused_bytes = _get_protocol_factory_for_bytes(bytes)
        protocol = protocol_factory(self.backing_transport, self._write_out, self.root_client_path)
        protocol.accept_bytes(unused_bytes)
        return protocol

    def _wait_on_descriptor(self, fd, timeout_seconds):
        """select() on a file descriptor, waiting for nonblocking read()

        This will raise a ConnectionTimeout exception if we do not get a
        readable handle before timeout_seconds.
        :return: None
        """
        t_end = self._timer() + timeout_seconds
        poll_timeout = min(timeout_seconds, self._client_poll_timeout)
        rs = xs = None
        while not rs and (not xs) and (self._timer() < t_end):
            if self.finished:
                return
            try:
                rs, _, xs = select.select([fd], [], [fd], poll_timeout)
            except OSError as e:
                err = getattr(e, 'errno', None)
                if err is None and getattr(e, 'args', None) is not None:
                    err = e.args[0]
                if err in _bad_file_descriptor:
                    return
                elif err == errno.EINTR:
                    continue
                raise
            except ValueError:
                return
        if rs or xs:
            return
        raise errors.ConnectionTimeout('disconnecting client after %.1f seconds' % (timeout_seconds,))

    def _serve_one_request(self, protocol):
        """Read one request from input, process, send back a response.

        :param protocol: a SmartServerRequestProtocol.
        """
        if protocol is None:
            return
        try:
            self._serve_one_request_unguarded(protocol)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.terminate_due_to_error()

    def terminate_due_to_error(self):
        """Called when an unhandled exception from the protocol occurs."""
        raise NotImplementedError(self.terminate_due_to_error)

    def _read_bytes(self, desired_count):
        """Get some bytes from the medium.

        :param desired_count: number of bytes we want to read.
        """
        raise NotImplementedError(self._read_bytes)