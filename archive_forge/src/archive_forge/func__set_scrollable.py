from __future__ import annotations
from collections import defaultdict, namedtuple
from typing import (
import param
from bokeh.models import Row as BkRow
from param.parameterized import iscoroutinefunction, resolve_ref
from ..io.document import freeze_doc
from ..io.model import hold
from ..io.resources import CDN_DIST
from ..io.state import state
from ..models import Column as PnColumn
from ..reactive import Reactive
from ..util import param_name, param_reprs, param_watchers
@param.depends('auto_scroll_limit', 'scroll_button_threshold', 'view_latest', watch=True, on_init=True)
def _set_scrollable(self):
    self.scroll = self.scroll or bool(self.scroll_position) or bool(self.auto_scroll_limit) or bool(self.scroll_button_threshold) or self.view_latest