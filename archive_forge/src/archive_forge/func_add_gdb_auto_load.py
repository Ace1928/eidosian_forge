from __future__ import annotations
import os, subprocess
import argparse
import tempfile
import shutil
import itertools
import typing as T
from pathlib import Path
from . import build, minstall
from .mesonlib import (EnvironmentVariables, MesonException, is_windows, setup_vsenv, OptionKey,
from . import mlog
def add_gdb_auto_load(autoload_path: Path, gdb_helper: str, fname: Path) -> None:
    destdir = autoload_path / fname.parent
    destdir.mkdir(parents=True, exist_ok=True)
    try:
        if is_windows():
            shutil.copy(gdb_helper, str(destdir / os.path.basename(gdb_helper)))
        else:
            os.symlink(gdb_helper, str(destdir / os.path.basename(gdb_helper)))
    except (FileExistsError, shutil.SameFileError):
        pass