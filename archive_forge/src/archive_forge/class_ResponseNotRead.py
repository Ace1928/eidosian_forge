from __future__ import annotations
import contextlib
import typing
class ResponseNotRead(StreamError):
    """
    Attempted to access streaming response content, without having called `read()`.
    """

    def __init__(self) -> None:
        message = 'Attempted to access streaming response content, without having called `read()`.'
        super().__init__(message)