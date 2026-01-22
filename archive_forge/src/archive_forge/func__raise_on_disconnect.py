from __future__ import annotations
import enum
import json
import typing
from starlette.requests import HTTPConnection
from starlette.types import Message, Receive, Scope, Send
def _raise_on_disconnect(self, message: Message) -> None:
    if message['type'] == 'websocket.disconnect':
        raise WebSocketDisconnect(message['code'], message.get('reason'))