from __future__ import annotations
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import ImportedStyleSheet
from bokeh.models.layouts import (
from .._param import Margin
from ..io.cache import _generate_hash
from ..io.document import create_doc_if_none_exists, unlocked
from ..io.notebook import push
from ..io.state import state
from ..layout.base import (
from ..links import Link
from ..models import ReactiveHTML as _BkReactiveHTML
from ..reactive import Reactive
from ..util import param_reprs, param_watchers
from ..util.checks import is_dataframe, is_series
from ..util.parameters import get_params_to_inherit
from ..viewable import (
@param.depends('_pane', '_pane.sizing_mode', '_pane.width_policy', '_pane.height_policy', watch=True)
def _sync_layout(self):
    if not hasattr(self, '_inner_layout') or (self._pane is not None and getattr(self._pane, '_object_changing', False)):
        return
    self._inner_layout.param.update({k: v for k, v in self._pane.param.values().items() if k in ('sizing_mode', 'width_policy', 'height_policy')})