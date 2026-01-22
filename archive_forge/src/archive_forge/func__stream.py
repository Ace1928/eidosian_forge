from __future__ import annotations
import logging # isort:skip
import copy
from typing import (
import numpy as np
def _stream(self, doc: Document, source: ColumnarDataSource, new_data: dict[str, Any], rollover: int | None=None, setter: Setter | None=None) -> None:
    """ Internal implementation to handle special-casing stream events
        on ``ColumnDataSource`` columns.

        Normally any changes to the ``.data`` dict attribute on a
        ``ColumnDataSource`` triggers a notification, causing all of the data
        to be synchronized between server and clients.

        The ``.stream`` method on column data sources exists to provide a
        more efficient way to perform streaming (i.e. append-only) updates
        to a data source, without having to perform a full synchronization,
        which would needlessly re-send all the data.

        To accomplish this, this function bypasses the wrapped methods on
        ``PropertyValueDict`` and uses the unwrapped versions on the dict
        superclass directly. It then explicitly makes a notification, adding
        a special ``ColumnsStreamedEvent`` hint to the message containing
        only the small streamed data that BokehJS needs in order to
        efficiently synchronize.

        .. warning::
            This function assumes the integrity of ``new_data`` has already
            been verified.

        """
    old = self._saved_copy()
    for k in new_data:
        if isinstance(self[k], np.ndarray) or isinstance(new_data[k], np.ndarray):
            data = np.append(self[k], new_data[k])
            if rollover is not None and len(data) > rollover:
                data = data[len(data) - rollover:]
            dict.__setitem__(self, k, data)
        else:
            L = self[k]
            L.extend(new_data[k])
            if rollover is not None and len(L) > rollover:
                del L[:len(L) - rollover]
    from ...document.events import ColumnsStreamedEvent
    self._notify_owners(old, hint=ColumnsStreamedEvent(doc, source, 'data', new_data, rollover, setter))