from __future__ import annotations
import json
from collections import UserString
from typing import Any, IO
class SphinxJSONEncoder(json.JSONEncoder):
    """JSONEncoder subclass that forces translation proxies."""

    def default(self, obj: Any) -> str:
        if isinstance(obj, UserString):
            return str(obj)
        return super().default(obj)