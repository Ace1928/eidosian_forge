from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
class _SmartClient:

    def __init__(self, medium, headers=None):
        """Constructor.

        :param medium: a SmartClientMedium
        """
        self._medium = medium
        if headers is None:
            self._headers = {b'Software version': breezy.__version__.encode('utf-8')}
        else:
            self._headers = dict(headers)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self._medium)

    def _call_and_read_response(self, method, args, body=None, readv_body=None, body_stream=None, expect_response_body=True):
        request = _SmartClientRequest(self, method, args, body=body, readv_body=readv_body, body_stream=body_stream, expect_response_body=expect_response_body)
        return request.call_and_read_response()

    def call(self, method, *args):
        """Call a method on the remote server."""
        result, protocol = self.call_expecting_body(method, *args)
        protocol.cancel_read_body()
        return result

    def call_expecting_body(self, method, *args):
        """Call a method and return the result and the protocol object.

        The body can be read like so::

            result, smart_protocol = smart_client.call_expecting_body(...)
            body = smart_protocol.read_body_bytes()
        """
        return self._call_and_read_response(method, args, expect_response_body=True)

    def call_with_body_bytes(self, method, args, body):
        """Call a method on the remote server with body bytes."""
        if not isinstance(method, bytes):
            raise TypeError('method must be a byte string, not {!r}'.format(method))
        for arg in args:
            if not isinstance(arg, bytes):
                raise TypeError('args must be byte strings, not {!r}'.format(args))
        if not isinstance(body, bytes):
            raise TypeError('body must be byte string, not {!r}'.format(body))
        response, response_handler = self._call_and_read_response(method, args, body=body, expect_response_body=False)
        return response

    def call_with_body_bytes_expecting_body(self, method, args, body):
        """Call a method on the remote server with body bytes."""
        if not isinstance(method, bytes):
            raise TypeError('method must be a byte string, not {!r}'.format(method))
        for arg in args:
            if not isinstance(arg, bytes):
                raise TypeError('args must be byte strings, not {!r}'.format(args))
        if not isinstance(body, bytes):
            raise TypeError('body must be byte string, not {!r}'.format(body))
        response, response_handler = self._call_and_read_response(method, args, body=body, expect_response_body=True)
        return (response, response_handler)

    def call_with_body_readv_array(self, args, body):
        response, response_handler = self._call_and_read_response(args[0], args[1:], readv_body=body, expect_response_body=True)
        return (response, response_handler)

    def call_with_body_stream(self, args, stream):
        response, response_handler = self._call_and_read_response(args[0], args[1:], body_stream=stream, expect_response_body=False)
        return (response, response_handler)

    def remote_path_from_transport(self, transport):
        """Convert transport into a path suitable for using in a request.

        Note that the resulting remote path doesn't encode the host name or
        anything but path, so it is only safe to use it in requests sent over
        the medium from the matching transport.
        """
        return self._medium.remote_path_from_transport(transport).encode('utf-8')