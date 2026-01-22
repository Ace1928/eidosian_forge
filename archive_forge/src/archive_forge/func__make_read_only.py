import json
import anywidget
import traitlets
import pathlib
from typing import Any, Set, Optional
import altair as alt
from altair.utils._vegafusion_data import (
from altair import TopLevelSpec
from altair.utils.selection import IndexSelection, PointSelection, IntervalSelection
def _make_read_only(self, change):
    """
        Work around to make traits read-only, but still allow us to change
        them internally
        """
    if change['name'] in self.traits() and change['old'] != change['new']:
        self._set_value(change['name'], change['old'])
    raise ValueError(f'Selections may not be set from Python.\nAttempted to set select: {change['name']}')