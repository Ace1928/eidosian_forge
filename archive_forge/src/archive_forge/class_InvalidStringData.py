from __future__ import annotations
class InvalidStringData(BSONError):
    """Raised when trying to encode a string containing non-UTF8 data."""