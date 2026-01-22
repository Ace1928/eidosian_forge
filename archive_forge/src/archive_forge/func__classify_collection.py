from __future__ import annotations
import collections
import os
import re
import time
import typing as t
from ..target import (
from ..util import (
from .python import (
from .csharp import (
from .powershell import (
from ..config import (
from ..metadata import (
from ..data import (
def _classify_collection(self, path: str) -> t.Optional[dict[str, str]]:
    """Return the classification for the given path using rules specific to collections."""
    result = self._classify_common(path)
    if result is not None:
        return result
    filename = os.path.basename(path)
    dummy, ext = os.path.splitext(filename)
    minimal: dict[str, str] = {}
    if path.startswith('changelogs/'):
        return minimal
    if path.startswith('docs/'):
        return minimal
    if '/' not in path:
        if path in ('.gitignore', 'COPYING', 'LICENSE', 'Makefile'):
            return minimal
        if ext in ('.in', '.md', '.rst', '.toml', '.txt'):
            return minimal
    return None