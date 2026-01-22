from __future__ import annotations
import contextlib
import typing
class CloseError(NetworkError):
    """
    Failed to close a connection.
    """