from __future__ import annotations
import contextlib
import typing
class WriteTimeout(TimeoutException):
    """
    Timed out while sending data to the host.
    """