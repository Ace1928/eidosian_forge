import asyncio
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
from .http_parser import RawResponseMessage
from .typedefs import LooseHeaders
class ServerDisconnectedError(ServerConnectionError):
    """Server disconnected."""

    def __init__(self, message: Union[RawResponseMessage, str, None]=None) -> None:
        if message is None:
            message = 'Server disconnected'
        self.args = (message,)
        self.message = message