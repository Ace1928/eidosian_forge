from __future__ import annotations
import functools
import typing
from debugpy import _version
@_api(cancelable=True)
def wait_for_client() -> None:
    """If there is a client connected to the debug adapter that is
    debugging this process, returns immediately. Otherwise, blocks
    until a client connects to the adapter.

    While this function is waiting, it can be canceled by calling
    `wait_for_client.cancel()` from another thread.
    """