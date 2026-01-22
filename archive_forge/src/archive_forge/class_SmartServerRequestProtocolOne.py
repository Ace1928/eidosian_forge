import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
class SmartServerRequestProtocolOne(SmartProtocolBase):
    """Server-side encoding and decoding logic for smart version 1."""

    def __init__(self, backing_transport, write_func, root_client_path='/', jail_root=None):
        self._backing_transport = backing_transport
        self._root_client_path = root_client_path
        self._jail_root = jail_root
        self.unused_data = b''
        self._finished = False
        self.in_buffer = b''
        self._has_dispatched = False
        self.request = None
        self._body_decoder = None
        self._write_func = write_func

    def accept_bytes(self, data):
        """Take bytes, and advance the internal state machine appropriately.

        :param data: must be a byte string
        """
        if not isinstance(data, bytes):
            raise ValueError(data)
        self.in_buffer += data
        if not self._has_dispatched:
            if b'\n' not in self.in_buffer:
                return
            self._has_dispatched = True
            try:
                first_line, self.in_buffer = self.in_buffer.split(b'\n', 1)
                first_line += b'\n'
                req_args = _decode_tuple(first_line)
                self.request = request.SmartServerRequestHandler(self._backing_transport, commands=request.request_handlers, root_client_path=self._root_client_path, jail_root=self._jail_root)
                self.request.args_received(req_args)
                if self.request.finished_reading:
                    self.unused_data = self.in_buffer
                    self.in_buffer = b''
                    self._send_response(self.request.response)
            except KeyboardInterrupt:
                raise
            except errors.UnknownSmartMethod as err:
                protocol_error = errors.SmartProtocolError("bad request '{}'".format(err.verb.decode('ascii')))
                failure = request.FailedSmartServerResponse((b'error', str(protocol_error).encode('utf-8')))
                self._send_response(failure)
                return
            except Exception as exception:
                log_exception_quietly()
                self._send_response(request.FailedSmartServerResponse((b'error', str(exception).encode('utf-8'))))
                return
        if self._has_dispatched:
            if self._finished:
                self.unused_data += self.in_buffer
                self.in_buffer = b''
                return
            if self._body_decoder is None:
                self._body_decoder = LengthPrefixedBodyDecoder()
            self._body_decoder.accept_bytes(self.in_buffer)
            self.in_buffer = self._body_decoder.unused_data
            body_data = self._body_decoder.read_pending_data()
            self.request.accept_body(body_data)
            if self._body_decoder.finished_reading:
                self.request.end_of_body()
                if not self.request.finished_reading:
                    raise AssertionError('no more body, request not finished')
            if self.request.response is not None:
                self._send_response(self.request.response)
                self.unused_data = self.in_buffer
                self.in_buffer = b''
            elif self.request.finished_reading:
                raise AssertionError('no response and we have finished reading.')

    def _send_response(self, response):
        """Send a smart server response down the output stream."""
        if self._finished:
            raise AssertionError('response already sent')
        args = response.args
        body = response.body
        self._finished = True
        self._write_protocol_version()
        self._write_success_or_failure_prefix(response)
        self._write_func(_encode_tuple(args))
        if body is not None:
            if not isinstance(body, bytes):
                raise ValueError(body)
            data = self._encode_bulk_data(body)
            self._write_func(data)

    def _write_protocol_version(self):
        """Write any prefixes this protocol requires.

        Version one doesn't send protocol versions.
        """

    def _write_success_or_failure_prefix(self, response):
        """Write the protocol specific success/failure prefix.

        For SmartServerRequestProtocolOne this is omitted but we
        call is_successful to ensure that the response is valid.
        """
        response.is_successful()

    def next_read_size(self):
        if self._finished:
            return 0
        if self._body_decoder is None:
            return 1
        else:
            return self._body_decoder.next_read_size()