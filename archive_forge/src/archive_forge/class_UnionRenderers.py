from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class UnionRenderers(SelectionPolicy):
    """
    When a data source is shared between multiple renderers, selecting a point on
    from any renderer will cause that row in the data source to be selected. The
    selection is made from the union of hit test results from all renderers.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)