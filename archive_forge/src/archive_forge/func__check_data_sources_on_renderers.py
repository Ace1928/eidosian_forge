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
@error(NON_MATCHING_DATA_SOURCES_ON_LEGEND_ITEM_RENDERERS)
def _check_data_sources_on_renderers(self):
    if isinstance(self.label, Field):
        if len({r.data_source for r in self.renderers}) != 1:
            return str(self)