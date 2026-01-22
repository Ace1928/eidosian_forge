from __future__ import annotations
from typing import (
import param
from bokeh.models import Column as BkColumn, CustomJS
from ..reactive import Reactive
from .base import NamedListPanel
from .card import Card
def _apply_style(self, i):
    if i == 0:
        margin = (5, 5, 0, 5)
    elif i == len(self) - 1:
        margin = (0, 5, 5, 5)
    else:
        margin = (0, 5, 0, 5)
    return dict(margin=margin, collapsed=i not in self.active)