from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class SessionCallbackRemoved(DocumentChangedEvent):
    """ A concrete event representing a change to remove an existing callback
    (e.g. periodic, timeout, or "next tick") from a Document.


    """

    def __init__(self, document: Document, callback: SessionCallback) -> None:
        """

        Args:
            document (Document) :
                A Bokeh document that is to be updated.

            callback (SessionCallback) :
                The callback to remove

        """
        super().__init__(document)
        self.callback = callback

    def dispatch(self, receiver: Any) -> None:
        """ Dispatch handling of this event to a receiver.

        This method will invoke ``receiver._session_callback_removed`` if
        it exists.

        """
        super().dispatch(receiver)
        if hasattr(receiver, '_session_callback_removed'):
            cast(SessionCallbackRemovedMixin, receiver)._session_callback_removed(self)