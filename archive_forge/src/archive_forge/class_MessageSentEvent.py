from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class MessageSentEvent(DocumentPatchedEvent):
    """

    """
    kind = 'MessageSent'

    def __init__(self, document: Document, msg_type: str, msg_data: Any | bytes, setter: Setter | None=None, callback_invoker: Invoker | None=None):
        super().__init__(document, setter, callback_invoker)
        self.msg_type = msg_type
        self.msg_data = msg_data

    def dispatch(self, receiver: Any) -> None:
        super().dispatch(receiver)
        if hasattr(receiver, '_document_message_sent'):
            cast(DocumentMessageSentMixin, receiver)._document_message_sent(self)

    def to_serializable(self, serializer: Serializer) -> MessageSent:
        return MessageSent(kind=self.kind, msg_type=self.msg_type, msg_data=serializer.encode(self.msg_data))

    @staticmethod
    def _handle_event(doc: Document, event: MessageSentEvent) -> None:
        message_callbacks = doc.callbacks._message_callbacks.get(event.msg_type, [])
        for cb in message_callbacks:
            cb(event.msg_data)