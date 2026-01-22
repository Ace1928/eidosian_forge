from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
class ConventionalRequestHandler(MessageHandler):
    """A message handler for "conventional" requests.

    "Conventional" is used in the sense described in
    doc/developers/network-protocol.txt: a simple message with arguments and an
    optional body.

    Possible states:
     * args: expecting args
     * body: expecting body (terminated by receiving a post-body status)
     * error: expecting post-body error
     * end: expecting end of message
     * nothing: finished
    """

    def __init__(self, request_handler, responder):
        MessageHandler.__init__(self)
        self.request_handler = request_handler
        self.responder = responder
        self.expecting = 'args'
        self._should_finish_body = False
        self._response_sent = False

    def protocol_error(self, exception):
        if self.responder.response_sent:
            return
        self.responder.send_error(exception)

    def byte_part_received(self, byte):
        if not isinstance(byte, bytes):
            raise TypeError(byte)
        if self.expecting == 'body':
            if byte == b'S':
                self.expecting = 'end'
            elif byte == b'E':
                self.expecting = 'error'
            else:
                raise errors.SmartProtocolError('Non-success status byte in request body: {!r}'.format(byte))
        else:
            raise errors.SmartProtocolError('Unexpected message part: byte({!r})'.format(byte))

    def structure_part_received(self, structure):
        if self.expecting == 'args':
            self._args_received(structure)
        elif self.expecting == 'error':
            self._error_received(structure)
        else:
            raise errors.SmartProtocolError('Unexpected message part: structure({!r})'.format(structure))

    def _args_received(self, args):
        self.expecting = 'body'
        self.request_handler.args_received(args)
        if self.request_handler.finished_reading:
            self._response_sent = True
            self.responder.send_response(self.request_handler.response)
            self.expecting = 'end'

    def _error_received(self, error_args):
        self.expecting = 'end'
        self.request_handler.post_body_error_received(error_args)

    def bytes_part_received(self, bytes):
        if self.expecting == 'body':
            self._should_finish_body = True
            self.request_handler.accept_body(bytes)
        else:
            raise errors.SmartProtocolError('Unexpected message part: bytes({!r})'.format(bytes))

    def end_received(self):
        if self.expecting not in ['body', 'end']:
            raise errors.SmartProtocolError('End of message received prematurely (while expecting %s)' % (self.expecting,))
        self.expecting = 'nothing'
        self.request_handler.end_received()
        if not self.request_handler.finished_reading:
            raise errors.SmartProtocolError('Complete conventional request was received, but request handler has not finished reading.')
        if not self._response_sent:
            self.responder.send_response(self.request_handler.response)