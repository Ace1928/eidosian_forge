from __future__ import annotations
import logging # isort:skip
from typing import (
import numpy as np
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from ..util.deprecation import deprecated
from ..util.serialization import convert_datetime_array
from ..util.warnings import BokehUserWarning, warn
from .callbacks import CustomJS
from .filters import AllIndices, Filter, IntersectionFilter
from .selections import Selection, SelectionPolicy, UnionRenderers
class CDSView(Model):
    """ A view into a ``ColumnDataSource`` that represents a row-wise subset.

    """

    def __init__(self, *args: TAny, **kwargs: TAny) -> None:
        if 'source' in kwargs:
            del kwargs['source']
            deprecated('CDSView.source is no longer needed, and is now ignored. In a future release, passing source will result an error.')
        super().__init__(*args, **kwargs)
    filter = Instance(Filter, default=InstanceDefault(AllIndices), help='\n    Defines the subset of indices to use from the data source this view applies to.\n\n    By default all indices are used (``AllIndices`` filter). This can be changed by\n    using specialized filters like ``IndexFilter``, ``BooleanFilter``, etc. Filters\n    can be composed using set operations to create non-trivial data masks. This can\n    be accomplished by directly using models like ``InversionFilter``, ``UnionFilter``,\n    etc., or by using set operators on filters, e.g.:\n\n    .. code-block:: python\n\n        # filters everything but indexes 10 and 11\n        cds_view.filter &= ~IndexFilter(indices=[10, 11])\n    ')

    @property
    def filters(self) -> list[Filter]:
        deprecated('CDSView.filters was deprecated in bokeh 3.0. Use CDSView.filter instead.')
        filter = self.filter
        if isinstance(filter, IntersectionFilter):
            return filter.operands
        elif isinstance(filter, AllIndices):
            return []
        else:
            return [filter]

    @filters.setter
    def filters(self, filters: list[Filter]) -> None:
        deprecated('CDSView.filters was deprecated in bokeh 3.0. Use CDSView.filter instead.')
        if len(filters) == 0:
            self.filter = AllIndices()
        elif len(filters) == 1:
            self.filter = filters[0]
        else:
            self.filter = IntersectionFilter(operands=filters)