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
def _get_args_for_reloading() -> list[str]:
    """Determine how the script was executed, and return the args needed
    to execute it again in a new process.
    """
    if sys.version_info >= (3, 10):
        return [sys.executable, *sys.orig_argv[1:]]
    rv = [sys.executable]
    py_script = sys.argv[0]
    args = sys.argv[1:]
    __main__ = sys.modules['__main__']
    if getattr(__main__, '__package__', None) is None or (os.name == 'nt' and __main__.__package__ == '' and (not os.path.exists(py_script)) and os.path.exists(f'{py_script}.exe')):
        py_script = os.path.abspath(py_script)
        if os.name == 'nt':
            if not os.path.exists(py_script) and os.path.exists(f'{py_script}.exe'):
                py_script += '.exe'
            if os.path.splitext(sys.executable)[1] == '.exe' and os.path.splitext(py_script)[1] == '.exe':
                rv.pop(0)
        rv.append(py_script)
    else:
        if os.path.isfile(py_script):
            py_module = t.cast(str, __main__.__package__)
            name = os.path.splitext(os.path.basename(py_script))[0]
            if name != '__main__':
                py_module += f'.{name}'
        else:
            py_module = py_script
        rv.extend(('-m', py_module.lstrip('.')))
    rv.extend(args)
    return rv