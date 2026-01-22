from __future__ import annotations
import contextlib
import typing
class WriteError(NetworkError):
    """
    Failed to send data through the network.
    """