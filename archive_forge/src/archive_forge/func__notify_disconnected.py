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
def _notify_disconnected(self) -> None:
    """ Called by the ClientConnection we are using to notify us of disconnect.

        """
    if self.document is not None:
        self.document.remove_on_change(self)
        self._callbacks.remove_all_callbacks()