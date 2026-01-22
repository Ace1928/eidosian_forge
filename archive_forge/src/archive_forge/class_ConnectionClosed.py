from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class ConnectionClosed(WebSocketException):
    """
    Raised when trying to interact with a closed connection.

    Attributes:
        rcvd (Optional[Close]): if a close frame was received, its code and
            reason are available in ``rcvd.code`` and ``rcvd.reason``.
        sent (Optional[Close]): if a close frame was sent, its code and reason
            are available in ``sent.code`` and ``sent.reason``.
        rcvd_then_sent (Optional[bool]): if close frames were received and
            sent, this attribute tells in which order this happened, from the
            perspective of this side of the connection.

    """

    def __init__(self, rcvd: Optional[frames.Close], sent: Optional[frames.Close], rcvd_then_sent: Optional[bool]=None) -> None:
        self.rcvd = rcvd
        self.sent = sent
        self.rcvd_then_sent = rcvd_then_sent

    def __str__(self) -> str:
        if self.rcvd is None:
            if self.sent is None:
                assert self.rcvd_then_sent is None
                return 'no close frame received or sent'
            else:
                assert self.rcvd_then_sent is None
                return f'sent {self.sent}; no close frame received'
        elif self.sent is None:
            assert self.rcvd_then_sent is None
            return f'received {self.rcvd}; no close frame sent'
        else:
            assert self.rcvd_then_sent is not None
            if self.rcvd_then_sent:
                return f'received {self.rcvd}; then sent {self.sent}'
            else:
                return f'sent {self.sent}; then received {self.rcvd}'

    @property
    def code(self) -> int:
        if self.rcvd is None:
            return frames.CloseCode.ABNORMAL_CLOSURE
        return self.rcvd.code

    @property
    def reason(self) -> str:
        if self.rcvd is None:
            return ''
        return self.rcvd.reason