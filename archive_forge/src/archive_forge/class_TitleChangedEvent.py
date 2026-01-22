from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class TitleChangedEvent(DocumentPatchedEvent):
    """ A concrete event representing a change to the title of a Bokeh
    Document.

    """
    kind = 'TitleChanged'

    def __init__(self, document: Document, title: str, setter: Setter | None=None, callback_invoker: Invoker | None=None):
        """

        Args:
            document (Document) :
                A Bokeh document that is to be updated.

            title (str) :
                The new title to set on the Document

            setter (ClientSession or ServerSession or None, optional) :
                This is used to prevent "boomerang" updates to Bokeh apps.
                (default: None)

                See :class:`~bokeh.document.events.DocumentChangedEvent`
                for more details.

            callback_invoker (callable, optional) :
                A callable that will invoke any Model callbacks that should
                be executed in response to the change that triggered this
                event. (default: None)


        """
        super().__init__(document, setter, callback_invoker)
        self.title = title

    def combine(self, event: DocumentChangedEvent) -> bool:
        """

        """
        if not isinstance(event, TitleChangedEvent):
            return False
        if self.setter != event.setter:
            return False
        if self.document != event.document:
            return False
        self.title = event.title
        self.callback_invoker = event.callback_invoker
        return True

    def to_serializable(self, serializer: Serializer) -> TitleChanged:
        """ Create a JSON representation of this event suitable for sending
        to clients.

        .. code-block:: python

            {
                'kind'  : 'TitleChanged'
                'title' : <new title to set>
            }

        Args:
            serializer (Serializer):

        """
        return TitleChanged(kind=self.kind, title=self.title)

    @staticmethod
    def _handle_event(doc: Document, event: TitleChangedEvent) -> None:
        doc.set_title(event.title, event.setter)