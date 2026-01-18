from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
 Creates a scatter plot of the given x and y items.

        Args:
            x (str or seq[float]) : values or field names of center x coordinates

            y (str or seq[float]) : values or field names of center y coordinates

            size (str or list[float]) : values or field names of sizes in |screen units|

            marker (str, or list[str]): values or field names of marker types

            color (color value, optional): shorthand to set both fill and line color

            source (:class:`~bokeh.models.sources.ColumnDataSource`) : a user-supplied data source.
                An attempt will be made to convert the object to :class:`~bokeh.models.sources.ColumnDataSource`
                if needed. If none is supplied, one is created for the user automatically.

            **kwargs: |line properties| and |fill properties|

        Examples:

            >>> p.scatter([1,2,3],[4,5,6], marker="square", fill_color="red")
            >>> p.scatter("data1", "data2", marker="mtype", source=data_source, ...)

        .. note::
            ``Scatter`` markers with multiple marker types may be drawn in a
            different order when using the WebGL output backend. This is an explicit
            trade-off made in the interests of performance.

        