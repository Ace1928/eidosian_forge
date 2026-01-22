from __future__ import annotations
import dataclasses
import urllib.parse
from typing import Optional, Tuple
from . import exceptions
@dataclasses.dataclass
class WebSocketURI:
    """
    WebSocket URI.

    Attributes:
        secure: :obj:`True` for a ``wss`` URI, :obj:`False` for a ``ws`` URI.
        host: Normalized to lower case.
        port: Always set even if it's the default.
        path: May be empty.
        query: May be empty if the URI doesn't include a query component.
        username: Available when the URI contains `User Information`_.
        password: Available when the URI contains `User Information`_.

    .. _User Information: https://www.rfc-editor.org/rfc/rfc3986.html#section-3.2.1

    """
    secure: bool
    host: str
    port: int
    path: str
    query: str
    username: Optional[str] = None
    password: Optional[str] = None

    @property
    def resource_name(self) -> str:
        if self.path:
            resource_name = self.path
        else:
            resource_name = '/'
        if self.query:
            resource_name += '?' + self.query
        return resource_name

    @property
    def user_info(self) -> Optional[Tuple[str, str]]:
        if self.username is None:
            return None
        assert self.password is not None
        return (self.username, self.password)