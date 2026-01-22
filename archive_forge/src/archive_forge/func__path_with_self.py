from __future__ import annotations
import argparse
import errno
import json
import os
import site
import sys
import sysconfig
from pathlib import Path
from shutil import which
from subprocess import Popen
from typing import Any
from . import paths
from .version import __version__
def _path_with_self() -> list[str]:
    """Put `jupyter`'s dir at the front of PATH

    Ensures that /path/to/jupyter subcommand
    will do /path/to/jupyter-subcommand
    even if /other/jupyter-subcommand is ahead of it on PATH
    """
    path_list = (os.environ.get('PATH') or os.defpath).split(os.pathsep)
    try:
        bindir = sysconfig.get_path('scripts')
    except KeyError:
        pass
    else:
        path_list.append(bindir)
    scripts = [sys.argv[0]]
    if Path(scripts[0]).is_symlink():
        scripts.append(os.path.realpath(scripts[0]))
    for script in scripts:
        bindir = str(Path(script).parent)
        if Path(bindir).is_dir() and os.access(script, os.X_OK):
            path_list.insert(0, bindir)
    return path_list