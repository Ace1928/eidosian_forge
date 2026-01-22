from __future__ import annotations
import os
import re
import sys
from functools import lru_cache
from typing import cast
from .api import PlatformDirsABC
@lru_cache(maxsize=1)
def _android_documents_folder() -> str:
    """:return: documents folder for the Android OS"""
    try:
        from jnius import autoclass
        context = autoclass('android.content.Context')
        environment = autoclass('android.os.Environment')
        documents_dir: str = context.getExternalFilesDir(environment.DIRECTORY_DOCUMENTS).getAbsolutePath()
    except Exception:
        documents_dir = '/storage/emulated/0/Documents'
    return documents_dir