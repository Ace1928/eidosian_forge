from __future__ import annotations
import contextlib
import typing
class StreamClosed(StreamError):
    """
    Attempted to read or stream response content, but the request has been
    closed.
    """

    def __init__(self) -> None:
        message = 'Attempted to read or stream content, but the stream has been closed.'
        super().__init__(message)