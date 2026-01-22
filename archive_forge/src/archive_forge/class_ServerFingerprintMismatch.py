import asyncio
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
from .http_parser import RawResponseMessage
from .typedefs import LooseHeaders
class ServerFingerprintMismatch(ServerConnectionError):
    """SSL certificate does not match expected fingerprint."""

    def __init__(self, expected: bytes, got: bytes, host: str, port: int) -> None:
        self.expected = expected
        self.got = got
        self.host = host
        self.port = port
        self.args = (expected, got, host, port)

    def __repr__(self) -> str:
        return '<{} expected={!r} got={!r} host={!r} port={!r}>'.format(self.__class__.__name__, self.expected, self.got, self.host, self.port)