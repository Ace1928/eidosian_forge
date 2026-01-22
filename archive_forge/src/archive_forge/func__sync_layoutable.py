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
def _sync_layoutable(self, *events: param.parameterized.Event):
    included = set(Layoutable.param) - set(self._skip_layoutable)
    if events:
        kwargs = {event.name: event.new for event in events if event.name in included}
    else:
        kwargs = {k: v for k, v in self.param.values().items() if k in included}
    if self.margin:
        margin = self.margin
        if isinstance(margin, tuple):
            if len(margin) == 2:
                t = b = margin[0]
                r = l = margin[1]
            else:
                t, r, b, l = margin
        else:
            t = r = b = l = margin
        if kwargs.get('width') is not None:
            kwargs['width'] = kwargs['width'] + l + r
        if kwargs.get('height') is not None:
            kwargs['height'] = kwargs['height'] + t + b
    old_values = self.layout.param.values()
    self.layout.param.update({k: v for k, v in kwargs.items() if v != old_values[k]})