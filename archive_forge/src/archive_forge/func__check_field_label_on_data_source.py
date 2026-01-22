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
@error(BAD_COLUMN_NAME)
def _check_field_label_on_data_source(self):
    if isinstance(self.label, Field):
        if len(self.renderers) < 1:
            return str(self)
        source = self.renderers[0].data_source
        if self.label.field not in source.column_names:
            return str(self)