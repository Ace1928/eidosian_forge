from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
class _SmartClientRequest:
    """Encapsulate the logic for a single request.

    This class handles things like reconnecting and sending the request a
    second time when the connection is reset in the middle. It also handles the
    multiple requests that get made if we don't know what protocol the server
    supports yet.

    Generally, you build up one of these objects, passing in the arguments that
    you want to send to the server, and then use 'call_and_read_response' to
    get the response from the server.
    """

    def __init__(self, client, method, args, body=None, readv_body=None, body_stream=None, expect_response_body=True):
        self.client = client
        self.method = method
        self.args = args
        self.body = body
        self.readv_body = readv_body
        self.body_stream = body_stream
        self.expect_response_body = expect_response_body

    def call_and_read_response(self):
        """Send the request to the server, and read the initial response.

        This doesn't read all of the body content of the response, instead it
        returns (response_tuple, response_handler). response_tuple is the 'ok',
        or 'error' information, and 'response_handler' can be used to get the
        content stream out.
        """
        self._run_call_hooks()
        protocol_version = self.client._medium._protocol_version
        if protocol_version is None:
            return self._call_determining_protocol_version()
        else:
            return self._call(protocol_version)

    def _is_safe_to_send_twice(self):
        """Check if the current method is re-entrant safe."""
        if self.body_stream is not None or 'noretry' in debug.debug_flags:
            return False
        request_type = _mod_request.request_handlers.get_info(self.method)
        if request_type in ('read', 'idem', 'semi'):
            return True
        if request_type in ('semivfs', 'mutate', 'stream'):
            return False
        trace.mutter('Unknown request type: %s for method %s' % (request_type, self.method))
        return False

    def _run_call_hooks(self):
        if not _SmartClient.hooks['call']:
            return
        params = CallHookParams(self.method, self.args, self.body, self.readv_body, self.client._medium)
        for hook in _SmartClient.hooks['call']:
            hook(params)

    def _call(self, protocol_version):
        """We know the protocol version.

        So this just sends the request, and then reads the response. This is
        where the code will be to retry requests if the connection is closed.
        """
        response_handler = self._send(protocol_version)
        try:
            response_tuple = response_handler.read_response_tuple(expect_body=self.expect_response_body)
        except errors.ConnectionReset as e:
            self.client._medium.reset()
            if not self._is_safe_to_send_twice():
                raise
            trace.warning('ConnectionReset reading response for %r, retrying' % (self.method,))
            trace.log_exception_quietly()
            encoder, response_handler = self._construct_protocol(protocol_version)
            self._send_no_retry(encoder)
            response_tuple = response_handler.read_response_tuple(expect_body=self.expect_response_body)
        return (response_tuple, response_handler)

    def _call_determining_protocol_version(self):
        """Determine what protocol the remote server supports.

        We do this by placing a request in the most recent protocol, and
        handling the UnexpectedProtocolVersionMarker from the server.
        """
        last_err = None
        for protocol_version in [3, 2]:
            if protocol_version == 2:
                self.client._medium._remember_remote_is_before((1, 6))
            try:
                response_tuple, response_handler = self._call(protocol_version)
            except errors.UnexpectedProtocolVersionMarker as err:
                trace.warning('Server does not understand Bazaar network protocol %d, reconnecting.  (Upgrade the server to avoid this.)' % (protocol_version,))
                self.client._medium.disconnect()
                last_err = err
                continue
            except errors.ErrorFromSmartServer:
                self.client._medium._protocol_version = protocol_version
                raise
            else:
                self.client._medium._protocol_version = protocol_version
                return (response_tuple, response_handler)
        raise errors.SmartProtocolError('Server is not a Bazaar server: ' + str(last_err))

    def _construct_protocol(self, version):
        """Build the encoding stack for a given protocol version."""
        request = self.client._medium.get_request()
        if version == 3:
            request_encoder = protocol.ProtocolThreeRequester(request)
            response_handler = message.ConventionalResponseHandler()
            response_proto = protocol.ProtocolThreeDecoder(response_handler, expect_version_marker=True)
            response_handler.setProtoAndMediumRequest(response_proto, request)
        elif version == 2:
            request_encoder = protocol.SmartClientRequestProtocolTwo(request)
            response_handler = request_encoder
        else:
            request_encoder = protocol.SmartClientRequestProtocolOne(request)
            response_handler = request_encoder
        return (request_encoder, response_handler)

    def _send(self, protocol_version):
        """Encode the request, and send it to the server.

        This will retry a request if we get a ConnectionReset while sending the
        request to the server. (Unless we have a body_stream that we have
        already started consuming, since we can't restart body_streams)

        :return: response_handler as defined by _construct_protocol
        """
        encoder, response_handler = self._construct_protocol(protocol_version)
        try:
            self._send_no_retry(encoder)
        except errors.ConnectionReset as e:
            self.client._medium.reset()
            if 'noretry' in debug.debug_flags or (self.body_stream is not None and encoder.body_stream_started):
                raise
            trace.warning('ConnectionReset calling %r, retrying' % (self.method,))
            trace.log_exception_quietly()
            encoder, response_handler = self._construct_protocol(protocol_version)
            self._send_no_retry(encoder)
        return response_handler

    def _send_no_retry(self, encoder):
        """Just encode the request and try to send it."""
        encoder.set_headers(self.client._headers)
        if self.body is not None:
            if self.readv_body is not None:
                raise AssertionError('body and readv_body are mutually exclusive.')
            if self.body_stream is not None:
                raise AssertionError('body and body_stream are mutually exclusive.')
            encoder.call_with_body_bytes((self.method,) + self.args, self.body)
        elif self.readv_body is not None:
            if self.body_stream is not None:
                raise AssertionError('readv_body and body_stream are mutually exclusive.')
            encoder.call_with_body_readv_array((self.method,) + self.args, self.readv_body)
        elif self.body_stream is not None:
            encoder.call_with_body_stream((self.method,) + self.args, self.body_stream)
        else:
            encoder.call(self.method, *self.args)