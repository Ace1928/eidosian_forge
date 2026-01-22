from __future__ import annotations
import contextlib
import typing
class DecodingError(RequestError):
    """
    Decoding of the response failed, due to a malformed encoding.
    """