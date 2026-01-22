from __future__ import annotations
import fnmatch
import os
import subprocess
import sys
import threading
import time
import typing as t
from itertools import chain
from pathlib import PurePath
from ._internal import _log
def _iter_module_paths() -> t.Iterator[str]:
    """Find the filesystem paths associated with imported modules."""
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name is None or name.startswith(_ignore_always):
            continue
        while not os.path.isfile(name):
            old = name
            name = os.path.dirname(name)
            if name == old:
                break
        else:
            yield name