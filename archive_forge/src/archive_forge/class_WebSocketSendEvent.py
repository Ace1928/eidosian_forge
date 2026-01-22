import sys
from typing import (
class WebSocketSendEvent(TypedDict):
    type: Literal['websocket.send']
    bytes: Optional[bytes]
    text: Optional[str]