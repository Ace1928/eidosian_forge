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
def _update_names(self, event: param.parameterized.Event) -> None:
    if len(event.new) == len(self._names):
        return
    names = []
    for obj in event.new:
        if obj in event.old:
            index = event.old.index(obj)
            name = self._names[index]
        else:
            name = obj.name
        names.append(name)
    self._names = names