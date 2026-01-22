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
def _jupyter_abspath(subcommand: str) -> str:
    """This method get the abspath of a specified jupyter-subcommand with no
    changes on ENV.
    """
    search_path = os.pathsep.join(_path_with_self())
    jupyter_subcommand = f'jupyter-{subcommand}'
    abs_path = which(jupyter_subcommand, path=search_path)
    if abs_path is None:
        msg = f'\nJupyter command `{jupyter_subcommand}` not found.'
        raise Exception(msg)
    if not os.access(abs_path, os.X_OK):
        msg = f'\nJupyter command `{jupyter_subcommand}` is not executable.'
        raise Exception(msg)
    return abs_path