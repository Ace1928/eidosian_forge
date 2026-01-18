from __future__ import annotations
from typing import ClassVar, List
import param
from ..io.resources import CDN_DIST
from ..reactive import ReactiveHTML
from .base import ListLike

        Iterates over the Viewable and any potential children in the
        applying the Selector.

        Arguments
        ---------
        selector: type or callable or None
          The selector allows selecting a subset of Viewables by
          declaring a type or callable function to filter by.

        Returns
        -------
        viewables: list(Viewable)
        