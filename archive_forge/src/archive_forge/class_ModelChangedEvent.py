from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class ModelChangedEvent(DocumentPatchedEvent):
    """ A concrete event representing updating an attribute and value of a
    specific Bokeh Model.

    """
    kind = 'ModelChanged'

    def __init__(self, document: Document, model: Model, attr: str, new: Any, setter: Setter | None=None, callback_invoker: Invoker | None=None):
        """

        Args:
            document (Document) :
                A Bokeh document that is to be updated.

            model (Model) :
                A Model to update

            attr (str) :
                The name of the attribute to update on the model.

            new (object) :
                The new value of the attribute

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
        self.attr = attr
        self.new = new

    def combine(self, event: DocumentChangedEvent) -> bool:
        """

        """
        if not isinstance(event, ModelChangedEvent):
            return False
        if self.setter != event.setter:
            return False
        if self.document != event.document:
            return False
        if self.model == event.model and self.attr == event.attr:
            self.new = event.new
            self.callback_invoker = event.callback_invoker
            return True
        return False

    def dispatch(self, receiver: Any) -> None:
        """ Dispatch handling of this event to a receiver.

        This method will invoke ``receiver._document_model_changed`` if it exists.

        """
        super().dispatch(receiver)
        if hasattr(receiver, '_document_model_changed'):
            cast(DocumentModelChangedMixin, receiver)._document_model_changed(self)

    def to_serializable(self, serializer: Serializer) -> ModelChanged:
        """ Create a JSON representation of this event suitable for sending
        to clients.

        Args:
            serializer (Serializer):

        """
        return ModelChanged(kind=self.kind, model=self.model.ref, attr=self.attr, new=serializer.encode(self.new))

    @staticmethod
    def _handle_event(doc: Document, event: ModelChangedEvent) -> None:
        model = event.model
        attr = event.attr
        value = event.new
        model.set_from_json(attr, value, setter=event.setter)