from __future__ import annotations
import typing as t
from datetime import datetime
class BadHeader(BadSignature):
    """Raised if a signed header is invalid in some form. This only
    happens for serializers that have a header that goes with the
    signature.

    .. versionadded:: 0.24
    """

    def __init__(self, message: str, payload: t.Any | None=None, header: t.Any | None=None, original_error: Exception | None=None):
        super().__init__(message, payload)
        self.header: t.Any | None = header
        self.original_error: Exception | None = original_error