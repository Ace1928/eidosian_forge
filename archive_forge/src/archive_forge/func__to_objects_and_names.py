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
def _to_objects_and_names(self, items):
    objects, names = ([], [])
    for item in items:
        pane, name = self._to_object_and_name(item)
        objects.append(pane)
        names.append(name)
    return (objects, names)