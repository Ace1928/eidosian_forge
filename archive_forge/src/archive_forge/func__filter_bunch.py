from __future__ import annotations
import json
import urllib.request
import uuid
from typing import Callable
from urllib.parse import quote
def _filter_bunch(bunch, keyword, name, requires_token, function):
    new = Bunch()
    for key, value in bunch.items():
        if isinstance(value, TileProvider):
            if function is None:
                if _validate(value, keyword=keyword, name=name, requires_token=requires_token):
                    new[key] = value
            elif function(value):
                new[key] = value
        else:
            filtered = _filter_bunch(value, keyword=keyword, name=name, requires_token=requires_token, function=function)
            if filtered:
                new[key] = filtered
    return new