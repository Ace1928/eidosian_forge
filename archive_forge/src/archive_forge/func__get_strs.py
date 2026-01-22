from __future__ import absolute_import
import uuid
from typing import Any, Iterable
def _get_strs(obj: Any) -> Iterable[str]:
    if obj is None:
        yield ''
    elif isinstance(obj, object) and hasattr(obj, '__uuid__'):
        yield str(obj.__uuid__())
    elif isinstance(obj, dict):
        for k, v in obj.items():
            for x in _get_strs(k):
                yield x
            for x in _get_strs(v):
                yield x
    elif not isinstance(obj, str) and isinstance(obj, Iterable):
        for k in obj:
            for x in _get_strs(k):
                yield x
    else:
        yield str(type(obj))
        yield str(obj)