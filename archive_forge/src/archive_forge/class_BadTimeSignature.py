from __future__ import annotations
import typing as t
from datetime import datetime
class BadTimeSignature(BadSignature):
    """Raised if a time-based signature is invalid. This is a subclass
    of :class:`BadSignature`.
    """

    def __init__(self, message: str, payload: t.Any | None=None, date_signed: datetime | None=None):
        super().__init__(message, payload)
        self.date_signed = date_signed