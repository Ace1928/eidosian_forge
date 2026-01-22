from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas.util._decorators import (
from pandas.core.indexers.objects import (
from pandas.core.window.doc import (
from pandas.core.window.rolling import (
class ExpandingGroupby(BaseWindowGroupby, Expanding):
    """
    Provide a expanding groupby implementation.
    """
    _attributes = Expanding._attributes + BaseWindowGroupby._attributes

    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
        window_indexer = GroupbyIndexer(groupby_indices=self._grouper.indices, window_indexer=ExpandingIndexer)
        return window_indexer