from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
import contextlib
import difflib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Union
from . import compat
def _run_console_script(self, path: str, options: Dict[str, Any]) -> None:
    """Run a Python console application from within the process.

        Used for black, zimports

        """
    is_posix = os.name == 'posix'
    entrypoint_name = options['entrypoint']
    for entry in compat.importlib_metadata_get('console_scripts'):
        if entry.name == entrypoint_name:
            impl = entry
            break
    else:
        raise Exception(f'Could not find entrypoint console_scripts.{entrypoint_name}')
    cmdline_options_str = options.get('options', '')
    cmdline_options_list = shlex.split(cmdline_options_str, posix=is_posix) + [path]
    kw: Dict[str, Any] = {}
    if self.suppress_output:
        kw['stdout'] = kw['stderr'] = subprocess.DEVNULL
    subprocess.run([sys.executable, '-c', 'import %s; %s.%s()' % (impl.module, impl.module, impl.attr)] + cmdline_options_list, cwd=str(self.source_root), **kw)