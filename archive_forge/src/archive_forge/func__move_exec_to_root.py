from __future__ import annotations
import os
import errno
import shutil
import subprocess
import sys
from pathlib import Path
from ._backend import Backend
from string import Template
from itertools import chain
import warnings
def _move_exec_to_root(self, build_dir: Path):
    walk_dir = Path(build_dir) / self.meson_build_dir
    path_objects = chain(walk_dir.glob(f'{self.modulename}*.so'), walk_dir.glob(f'{self.modulename}*.pyd'))
    for path_object in path_objects:
        dest_path = Path.cwd() / path_object.name
        if dest_path.exists():
            dest_path.unlink()
        shutil.copy2(path_object, dest_path)
        os.remove(path_object)