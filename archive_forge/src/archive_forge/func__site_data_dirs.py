from __future__ import annotations
import os
import sys
from configparser import ConfigParser
from pathlib import Path
from typing import Iterator
from .api import PlatformDirsABC
@property
def _site_data_dirs(self) -> list[str]:
    path = os.environ.get('XDG_DATA_DIRS', '')
    if not path.strip():
        path = f'/usr/local/share{os.pathsep}/usr/share'
    return [self._append_app_name_and_version(p) for p in path.split(os.pathsep)]