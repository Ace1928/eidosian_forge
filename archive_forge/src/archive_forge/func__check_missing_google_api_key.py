from __future__ import annotations
import logging # isort:skip
from ..core.enums import MapType
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error, warning
from ..core.validation.errors import (
from ..core.validation.warnings import MISSING_RENDERERS
from ..model import Model
from ..models.ranges import Range1d
from .plots import Plot
@error(MISSING_GOOGLE_API_KEY)
def _check_missing_google_api_key(self):
    if self.api_key is None:
        return str(self)