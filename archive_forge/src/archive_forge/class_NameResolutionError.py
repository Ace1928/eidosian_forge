from __future__ import annotations
import socket
import typing
import warnings
from email.errors import MessageDefect
from http.client import IncompleteRead as httplib_IncompleteRead
class NameResolutionError(NewConnectionError):
    """Raised when host name resolution fails."""

    def __init__(self, host: str, conn: HTTPConnection, reason: socket.gaierror):
        message = f"Failed to resolve '{host}' ({reason})"
        super().__init__(conn, message)