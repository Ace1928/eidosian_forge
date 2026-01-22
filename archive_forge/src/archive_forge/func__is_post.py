from __future__ import annotations
from collections import namedtuple
import re
def _is_post(self) -> bool:
    """Is post."""
    return bool(self.post > 0)