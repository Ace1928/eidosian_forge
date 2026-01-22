from __future__ import annotations
from vine import transform
from .message import AsyncMessage
def _on_message_sent(self, orig_message, new_message):
    orig_message.id = new_message.id
    orig_message.md5 = new_message.md5
    return new_message