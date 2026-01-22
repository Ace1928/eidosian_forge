from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class ColumnsStreamedEvent(DocumentPatchedEvent):
    """ A concrete event representing efficiently streaming new data
    to a :class:`~bokeh.models.sources.ColumnDataSource`

    """
    kind = 'ColumnsStreamed'
    data: DataDict

    def __init__(self, document: Document, model: Model, attr: str, data: DataDict | pd.DataFrame, rollover: int | None=None, setter: Setter | None=None, callback_invoker: Invoker | None=None):
        """

        Args:
            document (Document) :
                A Bokeh document that is to be updated.

            column_source (ColumnDataSource) :
                The data source to stream new data to.

            data (dict or DataFrame) :
                New data to stream.

                If a DataFrame, will be stored as ``{c: df[c] for c in df.columns}``

            rollover (int, optional) :
                A rollover limit. If the data source columns exceed this
                limit, earlier values will be discarded to maintain the
                column length under the limit.

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
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data = {c: data[c] for c in data.columns}
        self.data = data
        self.rollover = rollover

    def dispatch(self, receiver: Any) -> None:
        """ Dispatch handling of this event to a receiver.

        This method will invoke ``receiver._columns_streamed`` if it exists.

        """
        super().dispatch(receiver)
        if hasattr(receiver, '_columns_streamed'):
            cast(ColumnsStreamedMixin, receiver)._columns_streamed(self)

    def to_serializable(self, serializer: Serializer) -> ColumnsStreamed:
        """ Create a JSON representation of this event suitable for sending
        to clients.

        .. code-block:: python

            {
                'kind'          : 'ColumnsStreamed'
                'column_source' : <reference to a CDS>
                'data'          : <new data to steam to column_source>
                'rollover'      : <rollover limit>
            }

        Args:
            serializer (Serializer):

        """
        return ColumnsStreamed(kind=self.kind, model=self.model.ref, attr=self.attr, data=serializer.encode(self.data), rollover=self.rollover)

    @staticmethod
    def _handle_event(doc: Document, event: ColumnsStreamedEvent) -> None:
        model = event.model
        attr = event.attr
        assert attr == 'data'
        data = event.data
        rollover = event.rollover
        model._stream(data, rollover, event.setter)