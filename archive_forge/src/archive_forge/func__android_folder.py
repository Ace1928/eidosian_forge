from __future__ import annotations
import os
import re
import sys
from functools import lru_cache
from typing import cast
from .api import PlatformDirsABC
@lru_cache(maxsize=1)
def _android_folder() -> str | None:
    """:return: base folder for the Android OS or None if it cannot be found"""
    try:
        from jnius import autoclass
        context = autoclass('android.content.Context')
        result: str | None = context.getFilesDir().getParentFile().getAbsolutePath()
    except Exception:
        pattern = re.compile('/data/(data|user/\\d+)/(.+)/files')
        for path in sys.path:
            if pattern.match(path):
                result = path.split('/files')[0]
                break
        else:
            result = None
    return result