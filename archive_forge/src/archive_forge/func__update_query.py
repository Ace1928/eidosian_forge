from __future__ import annotations
import json
import urllib.parse as urlparse
from typing import (
import param
from ..models.location import Location as _BkLocation
from ..reactive import Syncable
from ..util import edit_readonly, parse_query
from .document import create_doc_if_none_exists
from .state import state
def _update_query(self, *events: param.parameterized.Event, query: Optional[Dict[str, Any]]=None) -> None:
    if self._syncing:
        return
    serialized = query or {}
    for e in events:
        matches = [ps for o, ps, _, _ in self._synced if o in (e.cls, e.obj)]
        if not matches:
            continue
        owner = e.cls if e.obj is None else e.obj
        try:
            val = owner.param[e.name].serialize(e.new)
        except Exception:
            val = e.new
        if not isinstance(val, str):
            val = json.dumps(val)
        serialized[matches[0][e.name]] = val
    self._syncing = True
    try:
        self.update_query(**{k: v for k, v in serialized.items() if v is not None})
    finally:
        self._syncing = False