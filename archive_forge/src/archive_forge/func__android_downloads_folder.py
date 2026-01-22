from __future__ import annotations
import os
import re
import sys
from functools import lru_cache
from typing import cast
from .api import PlatformDirsABC
@lru_cache(maxsize=1)
def _android_downloads_folder() -> str:
    """:return: downloads folder for the Android OS"""
    try:
        from jnius import autoclass
        context = autoclass('android.content.Context')
        environment = autoclass('android.os.Environment')
        downloads_dir: str = context.getExternalFilesDir(environment.DIRECTORY_DOWNLOADS).getAbsolutePath()
    except Exception:
        downloads_dir = '/storage/emulated/0/Downloads'
    return downloads_dir