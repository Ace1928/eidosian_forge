from typing import (
from urllib.parse import urlparse
class Origin:

    def __init__(self, scheme: bytes, host: bytes, port: int) -> None:
        self.scheme = scheme
        self.host = host
        self.port = port

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Origin) and self.scheme == other.scheme and (self.host == other.host) and (self.port == other.port)

    def __str__(self) -> str:
        scheme = self.scheme.decode('ascii')
        host = self.host.decode('ascii')
        port = str(self.port)
        return f'{scheme}://{host}:{port}'