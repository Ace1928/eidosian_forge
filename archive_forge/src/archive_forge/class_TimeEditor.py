from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.singletons import Intrinsic
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
class TimeEditor(CellEditor):
    """ Spinner-based time cell editor.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)