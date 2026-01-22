from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote_plus
from ..core.types import ID
from ..document import Document
from ..resources import DEFAULT_SERVER_HTTP_URL, SessionCoordinates
from ..util.browser import NEW_PARAM, BrowserLike, BrowserTarget
from ..util.token import generate_jwt_token, generate_session_id
from .states import ErrorReason
from .util import server_url_for_websocket_url, websocket_url_for_server_url
def _loop_until_closed(self) -> None:
    """ Execute a blocking loop that runs and executes event callbacks
        until the connection is closed (e.g. by hitting Ctrl-C).

        This function is intended to facilitate testing ONLY.

        """
    self._connection.loop_until_closed()