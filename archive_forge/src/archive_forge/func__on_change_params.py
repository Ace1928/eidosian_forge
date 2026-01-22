import json
import anywidget
import traitlets
import pathlib
from typing import Any, Set, Optional
import altair as alt
from altair.utils._vegafusion_data import (
from altair import TopLevelSpec
from altair.utils.selection import IndexSelection, PointSelection, IntervalSelection
@traitlets.observe('_params')
def _on_change_params(self, change):
    for param_name, value in change.new.items():
        setattr(self.params, param_name, value)