from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class RootRemovedEvent(DocumentPatchedEvent):
    """ A concrete event representing a change to remove an existing Model
    from a Document's collection of "root" models.

    """
    kind = 'RootRemoved'

    def __init__(self, document: Document, model: Model, setter: Setter | None=None, callback_invoker: Invoker | None=None) -> None:
        """

        Args:
            document (Document) :
                A Bokeh document that is to be updated.

            model (Model) :
                The Bokeh Model to remove as a Document root.

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
        self.model = model

    def to_serializable(self, serializer: Serializer) -> RootRemoved:
        """ Create a JSON representation of this event suitable for sending
        to clients.

        .. code-block:: python

            {
                'kind'  : 'RootRemoved'
                'title' : <reference to a Model>
            }

        Args:
            serializer (Serializer):

        """
        return RootRemoved(kind=self.kind, model=self.model.ref)

    @staticmethod
    def _handle_event(doc: Document, event: RootRemovedEvent) -> None:
        model = event.model
        doc.remove_root(model, event.setter)