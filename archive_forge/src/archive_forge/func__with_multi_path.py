from __future__ import annotations
import os
import sys
from configparser import ConfigParser
from pathlib import Path
from .api import PlatformDirsABC
def _with_multi_path(self, path: str) -> str:
    path_list = path.split(os.pathsep)
    if not self.multipath:
        path_list = path_list[0:1]
    path_list = [self._append_app_name_and_version(os.path.expanduser(p)) for p in path_list]
    return os.pathsep.join(path_list)