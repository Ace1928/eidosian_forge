from __future__ import annotations
import logging # isort:skip
from typing import Any
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.vectorization import Field
from ...core.property_mixins import (
from ...core.validation import error
from ...core.validation.errors import (
from ...model import Model
from ..formatters import TickFormatter
from ..labeling import LabelingPolicy, NoOverlap
from ..mappers import ColorMapper
from ..ranges import Range
from ..renderers import GlyphRenderer
from ..tickers import FixedTicker, Ticker
from .annotation import Annotation
from .dimensional import Dimensional, MetricLength
class LegendItem(Model):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.label, str):
            self.label = value(self.label)
    label = NullStringSpec(help="\n    A label for this legend. Can be a string, or a column of a\n    ColumnDataSource. If ``label`` is a field, then it must\n    be in the renderers' data_source.\n    ")
    renderers = List(Instance(GlyphRenderer), help='\n    A list of the glyph renderers to draw in the legend. If ``label`` is a field,\n    then all data_sources of renderers must be the same.\n    ')
    index = Nullable(Int, help='\n    The column data index to use for drawing the representative items.\n\n    If None (the default), then Bokeh will automatically choose an index to\n    use. If the label does not refer to a data column name, this is typically\n    the first data point in the data source. Otherwise, if the label does\n    refer to a column name, the legend will have "groupby" behavior, and will\n    choose and display representative points from every "group" in the column.\n\n    If set to a number, Bokeh will use that number as the index in all cases.\n    ')
    visible = Bool(default=True, help='\n    Whether the legend item should be displayed. See\n    :ref:`ug_basic_annotations_legends_item_visibility` in the user guide.\n    ')

    @error(NON_MATCHING_DATA_SOURCES_ON_LEGEND_ITEM_RENDERERS)
    def _check_data_sources_on_renderers(self):
        if isinstance(self.label, Field):
            if len({r.data_source for r in self.renderers}) != 1:
                return str(self)

    @error(BAD_COLUMN_NAME)
    def _check_field_label_on_data_source(self):
        if isinstance(self.label, Field):
            if len(self.renderers) < 1:
                return str(self)
            source = self.renderers[0].data_source
            if self.label.field not in source.column_names:
                return str(self)