from __future__ import annotations
from .mixins import ImmutableDictMixin
from .mixins import UpdateDictMixin
from .. import http
def _set_cache_value(self, key, value, type):
    """Used internally by the accessor properties."""
    if type is bool:
        if value:
            self[key] = None
        else:
            self.pop(key, None)
    elif value is None:
        self.pop(key, None)
    elif value is True:
        self[key] = None
    elif type is not None:
        self[key] = type(value)
    else:
        self[key] = value