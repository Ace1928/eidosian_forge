from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
class H2StreamStateMachine(object):
    """
    A single HTTP/2 stream state machine.

    This stream object implements basically the state machine described in
    RFC 7540 section 5.1.

    :param stream_id: The stream ID of this stream. This is stored primarily
        for logging purposes.
    """

    def __init__(self, stream_id):
        self.state = StreamState.IDLE
        self.stream_id = stream_id
        self.client = None
        self.headers_sent = None
        self.trailers_sent = None
        self.headers_received = None
        self.trailers_received = None
        self.stream_closed_by = None

    def process_input(self, input_):
        """
        Process a specific input in the state machine.
        """
        if not isinstance(input_, StreamInputs):
            raise ValueError('Input must be an instance of StreamInputs')
        try:
            func, target_state = _transitions[self.state, input_]
        except KeyError:
            old_state = self.state
            self.state = StreamState.CLOSED
            raise ProtocolError('Invalid input %s in state %s' % (input_, old_state))
        else:
            previous_state = self.state
            self.state = target_state
            if func is not None:
                try:
                    return func(self, previous_state)
                except ProtocolError:
                    self.state = StreamState.CLOSED
                    raise
                except AssertionError as e:
                    self.state = StreamState.CLOSED
                    raise ProtocolError(e)
            return []

    def request_sent(self, previous_state):
        """
        Fires when a request is sent.
        """
        self.client = True
        self.headers_sent = True
        event = _RequestSent()
        return [event]

    def response_sent(self, previous_state):
        """
        Fires when something that should be a response is sent. This 'response'
        may actually be trailers.
        """
        if not self.headers_sent:
            if self.client is True or self.client is None:
                raise ProtocolError('Client cannot send responses.')
            self.headers_sent = True
            event = _ResponseSent()
        else:
            assert not self.trailers_sent
            self.trailers_sent = True
            event = _TrailersSent()
        return [event]

    def request_received(self, previous_state):
        """
        Fires when a request is received.
        """
        assert not self.headers_received
        assert not self.trailers_received
        self.client = False
        self.headers_received = True
        event = RequestReceived()
        event.stream_id = self.stream_id
        return [event]

    def response_received(self, previous_state):
        """
        Fires when a response is received. Also disambiguates between responses
        and trailers.
        """
        if not self.headers_received:
            assert self.client is True
            self.headers_received = True
            event = ResponseReceived()
        else:
            assert not self.trailers_received
            self.trailers_received = True
            event = TrailersReceived()
        event.stream_id = self.stream_id
        return [event]

    def data_received(self, previous_state):
        """
        Fires when data is received.
        """
        event = DataReceived()
        event.stream_id = self.stream_id
        return [event]

    def window_updated(self, previous_state):
        """
        Fires when a window update frame is received.
        """
        event = WindowUpdated()
        event.stream_id = self.stream_id
        return [event]

    def stream_half_closed(self, previous_state):
        """
        Fires when an END_STREAM flag is received in the OPEN state,
        transitioning this stream to a HALF_CLOSED_REMOTE state.
        """
        event = StreamEnded()
        event.stream_id = self.stream_id
        return [event]

    def stream_ended(self, previous_state):
        """
        Fires when a stream is cleanly ended.
        """
        self.stream_closed_by = StreamClosedBy.RECV_END_STREAM
        event = StreamEnded()
        event.stream_id = self.stream_id
        return [event]

    def stream_reset(self, previous_state):
        """
        Fired when a stream is forcefully reset.
        """
        self.stream_closed_by = StreamClosedBy.RECV_RST_STREAM
        event = StreamReset()
        event.stream_id = self.stream_id
        return [event]

    def send_new_pushed_stream(self, previous_state):
        """
        Fires on the newly pushed stream, when pushed by the local peer.

        No event here, but definitionally this peer must be a server.
        """
        assert self.client is None
        self.client = False
        self.headers_received = True
        return []

    def recv_new_pushed_stream(self, previous_state):
        """
        Fires on the newly pushed stream, when pushed by the remote peer.

        No event here, but definitionally this peer must be a client.
        """
        assert self.client is None
        self.client = True
        self.headers_sent = True
        return []

    def send_push_promise(self, previous_state):
        """
        Fires on the already-existing stream when a PUSH_PROMISE frame is sent.
        We may only send PUSH_PROMISE frames if we're a server.
        """
        if self.client is True:
            raise ProtocolError('Cannot push streams from client peers.')
        event = _PushedRequestSent()
        return [event]

    def recv_push_promise(self, previous_state):
        """
        Fires on the already-existing stream when a PUSH_PROMISE frame is
        received. We may only receive PUSH_PROMISE frames if we're a client.

        Fires a PushedStreamReceived event.
        """
        if not self.client:
            if self.client is None:
                msg = 'Idle streams cannot receive pushes'
            else:
                msg = 'Cannot receive pushed streams as a server'
            raise ProtocolError(msg)
        event = PushedStreamReceived()
        event.parent_stream_id = self.stream_id
        return [event]

    def send_end_stream(self, previous_state):
        """
        Called when an attempt is made to send END_STREAM in the
        HALF_CLOSED_REMOTE state.
        """
        self.stream_closed_by = StreamClosedBy.SEND_END_STREAM

    def send_reset_stream(self, previous_state):
        """
        Called when an attempt is made to send RST_STREAM in a non-closed
        stream state.
        """
        self.stream_closed_by = StreamClosedBy.SEND_RST_STREAM

    def reset_stream_on_error(self, previous_state):
        """
        Called when we need to forcefully emit another RST_STREAM frame on
        behalf of the state machine.

        If this is the first time we've done this, we should also hang an event
        off the StreamClosedError so that the user can be informed. We know
        it's the first time we've done this if the stream is currently in a
        state other than CLOSED.
        """
        self.stream_closed_by = StreamClosedBy.SEND_RST_STREAM
        error = StreamClosedError(self.stream_id)
        event = StreamReset()
        event.stream_id = self.stream_id
        event.error_code = ErrorCodes.STREAM_CLOSED
        event.remote_reset = False
        error._events = [event]
        raise error

    def recv_on_closed_stream(self, previous_state):
        """
        Called when an unexpected frame is received on an already-closed
        stream.

        An endpoint that receives an unexpected frame should treat it as
        a stream error or connection error with type STREAM_CLOSED, depending
        on the specific frame. The error handling is done at a higher level:
        this just raises the appropriate error.
        """
        raise StreamClosedError(self.stream_id)

    def send_on_closed_stream(self, previous_state):
        """
        Called when an attempt is made to send data on an already-closed
        stream.

        This essentially overrides the standard logic by throwing a
        more-specific error: StreamClosedError. This is a ProtocolError, so it
        matches the standard API of the state machine, but provides more detail
        to the user.
        """
        raise StreamClosedError(self.stream_id)

    def recv_push_on_closed_stream(self, previous_state):
        """
        Called when a PUSH_PROMISE frame is received on a full stop
        stream.

        If the stream was closed by us sending a RST_STREAM frame, then we
        presume that the PUSH_PROMISE was in flight when we reset the parent
        stream. Rathen than accept the new stream, we just reset it.
        Otherwise, we should call this a PROTOCOL_ERROR: pushing a stream on a
        naturally closed stream is a real problem because it creates a brand
        new stream that the remote peer now believes exists.
        """
        assert self.stream_closed_by is not None
        if self.stream_closed_by == StreamClosedBy.SEND_RST_STREAM:
            raise StreamClosedError(self.stream_id)
        else:
            raise ProtocolError('Attempted to push on closed stream.')

    def send_push_on_closed_stream(self, previous_state):
        """
        Called when an attempt is made to push on an already-closed stream.

        This essentially overrides the standard logic by providing a more
        useful error message. It's necessary because simply indicating that the
        stream is closed is not enough: there is now a new stream that is not
        allowed to be there. The only recourse is to tear the whole connection
        down.
        """
        raise ProtocolError('Attempted to push on closed stream.')

    def send_informational_response(self, previous_state):
        """
        Called when an informational header block is sent (that is, a block
        where the :status header has a 1XX value).

        Only enforces that these are sent *before* final headers are sent.
        """
        if self.headers_sent:
            raise ProtocolError('Information response after final response')
        event = _ResponseSent()
        return [event]

    def recv_informational_response(self, previous_state):
        """
        Called when an informational header block is received (that is, a block
        where the :status header has a 1XX value).
        """
        if self.headers_received:
            raise ProtocolError('Informational response after final response')
        event = InformationalResponseReceived()
        event.stream_id = self.stream_id
        return [event]

    def recv_alt_svc(self, previous_state):
        """
        Called when receiving an ALTSVC frame.

        RFC 7838 allows us to receive ALTSVC frames at any stream state, which
        is really absurdly overzealous. For that reason, we want to limit the
        states in which we can actually receive it. It's really only sensible
        to receive it after we've sent our own headers and before the server
        has sent its header block: the server can't guarantee that we have any
        state around after it completes its header block, and the server
        doesn't know what origin we're talking about before we've sent ours.

        For that reason, this function applies a few extra checks on both state
        and some of the little state variables we keep around. If those suggest
        an unreasonable situation for the ALTSVC frame to have been sent in,
        we quietly ignore it (as RFC 7838 suggests).

        This function is also *not* always called by the state machine. In some
        states (IDLE, RESERVED_LOCAL, CLOSED) we don't bother to call it,
        because we know the frame cannot be valid in that state (IDLE because
        the server cannot know what origin the stream applies to, CLOSED
        because the server cannot assume we still have state around,
        RESERVED_LOCAL because by definition if we're in the RESERVED_LOCAL
        state then *we* are the server).
        """
        if self.client is False:
            return []
        if self.headers_received:
            return []
        return [AlternativeServiceAvailable()]

    def send_alt_svc(self, previous_state):
        """
        Called when sending an ALTSVC frame on this stream.

        For consistency with the restrictions we apply on receiving ALTSVC
        frames in ``recv_alt_svc``, we want to restrict when users can send
        ALTSVC frames to the situations when we ourselves would accept them.

        That means: when we are a server, when we have received the request
        headers, and when we have not yet sent our own response headers.
        """
        if self.headers_sent:
            raise ProtocolError('Cannot send ALTSVC after sending response headers.')
        return