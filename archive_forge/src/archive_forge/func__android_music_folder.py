from __future__ import annotations
import os
import re
import sys
from functools import lru_cache
from typing import cast
from .api import PlatformDirsABC
@lru_cache(maxsize=1)
def _android_music_folder() -> str:
    """:return: music folder for the Android OS"""
    try:
        from jnius import autoclass
        context = autoclass('android.content.Context')
        environment = autoclass('android.os.Environment')
        music_dir: str = context.getExternalFilesDir(environment.DIRECTORY_MUSIC).getAbsolutePath()
    except Exception:
        music_dir = '/storage/emulated/0/Music'
    return music_dir