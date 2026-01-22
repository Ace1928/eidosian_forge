from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
from tornado.httpclient import HTTPClientError, HTTPRequest
from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketError, websocket_connect
from ..core.types import ID
from ..protocol import Protocol
from ..protocol.exceptions import MessageError, ProtocolError, ValidationError
from ..protocol.receiver import Receiver
from ..util.strings import format_url_query_arguments
from ..util.tornado import fixup_windows_event_loop_policy
from .states import (
from .websocket import WebSocketClientConnectionWrapper
def _send_message_wait_for_reply(self, message: Message[Any]) -> Message[Any] | None:
    waiter = WAITING_FOR_REPLY(message.header['msgid'])
    self._state = waiter
    send_result: list[None] = []

    async def handle_message(message: Message[Any], send_result: list[None]) -> None:
        result = await self.send_message(message)
        send_result.append(result)
    self._loop.add_callback(handle_message, message, send_result)

    def have_send_result_or_disconnected() -> bool:
        return len(send_result) > 0 or self._state != waiter
    self._loop_until(have_send_result_or_disconnected)

    def have_reply_or_disconnected() -> bool:
        return self._state != waiter or waiter.reply is not None
    self._loop_until(have_reply_or_disconnected)
    return waiter.reply