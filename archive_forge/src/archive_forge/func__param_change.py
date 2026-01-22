from __future__ import annotations
import sys
from collections import defaultdict
from functools import partial
from typing import (
import param
from bokeh.models import Range1d, Spacer as _BkSpacer
from bokeh.themes.theme import Theme
from packaging.version import Version
from param.parameterized import register_reference_transform
from param.reactive import bind
from ..io import state, unlocked
from ..layout import (
from ..viewable import Layoutable, Viewable
from ..widgets import Player
from .base import PaneBase, RerenderError, panel
from .plot import Bokeh, Matplotlib
from .plotly import Plotly
def _param_change(self, *events: param.parameterized.Event) -> None:
    if self._object_changing:
        return
    self._track_overrides(*(e for e in events if e.name in Layoutable.param))
    super()._param_change(*(e for e in events if e.name in self._overrides + ['css_classes']))