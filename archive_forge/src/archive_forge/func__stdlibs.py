from __future__ import annotations
import ast
import base64
import copy
import io
import pathlib
import pkgutil
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from html import escape
from textwrap import dedent
from typing import Any, Dict, List
import markdown
def _stdlibs():
    if sys.version_info[:2] >= (3, 10):
        return sys.stdlib_module_names
    env_dir = str(pathlib.Path(sys.executable).parent.parent)
    modules = list(sys.builtin_module_names)
    for m in pkgutil.iter_modules():
        mpath = getattr(m.module_finder, 'path', '')
        if mpath.startswith(env_dir) and 'site-packages' not in mpath:
            modules.append(m.name)
    return modules