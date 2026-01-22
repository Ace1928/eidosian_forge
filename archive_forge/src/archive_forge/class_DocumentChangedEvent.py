from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class DocumentChangedEvent:
    """ Base class for all internal events representing a change to a
    Bokeh Document.

    """
    document: Document
    setter: Setter | None
    callback_invoker: Invoker | None

    def __init__(self, document: Document, setter: Setter | None=None, callback_invoker: Invoker | None=None) -> None:
        """

        Args:
            document (Document) :
                A Bokeh document that is to be updated.

            setter (ClientSession or ServerSession or None, optional) :
                This is used to prevent "boomerang" updates to Bokeh apps.
                (default: None)

                In the context of a Bokeh server application, incoming updates
                to properties will be annotated with the session that is
                doing the updating. This value is propagated through any
                subsequent change notifications that the update triggers.
                The session can compare the event setter to itself, and
                suppress any updates that originate from itself.

            callback_invoker (callable, optional) :
                A callable that will invoke any Model callbacks that should
                be executed in response to the change that triggered this
                event. (default: None)

        """
        self.document = document
        self.setter = setter
        self.callback_invoker = callback_invoker

    def combine(self, event: DocumentChangedEvent) -> bool:
        """

        """
        return False

    def dispatch(self, receiver: Any) -> None:
        """ Dispatch handling of this event to a receiver.

        This method will invoke ``receiver._document_changed`` if it exists.

        """
        if hasattr(receiver, '_document_changed'):
            cast(DocumentChangedMixin, receiver)._document_changed(self)