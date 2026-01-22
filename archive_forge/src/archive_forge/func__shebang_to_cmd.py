from __future__ import annotations
import functools
import os
import shutil
import stat
import sys
import re
import typing as T
from pathlib import Path
from . import mesonlib
from . import mlog
from .mesonlib import MachineChoice, OrderedSet
@staticmethod
def _shebang_to_cmd(script: str) -> T.Optional[T.List[str]]:
    """
        Check if the file has a shebang and manually parse it to figure out
        the interpreter to use. This is useful if the script is not executable
        or if we're on Windows (which does not understand shebangs).
        """
    try:
        with open(script, encoding='utf-8') as f:
            first_line = f.readline().strip()
        if first_line.startswith('#!'):
            commands = first_line[2:].split('#')[0].strip().split(maxsplit=1)
            if mesonlib.is_windows():
                if commands[0].startswith('/'):
                    commands[0] = commands[0].split('/')[-1]
                if len(commands) > 0 and commands[0] == 'env':
                    commands = commands[1:]
                if len(commands) > 0 and commands[0] == 'python3':
                    commands = mesonlib.python_command + commands[1:]
            elif mesonlib.is_haiku():
                if commands[0] == '/usr/bin/env':
                    commands = commands[1:]
                if len(commands) > 0 and commands[0] == 'python3':
                    commands = mesonlib.python_command + commands[1:]
            elif commands[0] == '/usr/bin/env' and commands[1] == 'python3':
                commands = mesonlib.python_command + commands[2:]
            elif commands[0].split('/')[-1] == 'python3':
                commands = mesonlib.python_command + commands[1:]
            return commands + [script]
    except Exception as e:
        mlog.debug(str(e))
    mlog.debug(f'Unusable script {script!r}')
    return None