from __future__ import annotations
import os
import typing as T
from ...mesonlib import EnvironmentException
@classmethod
def _unix_args_to_native(cls, args: T.List[str], info: MachineInfo) -> T.List[str]:
    result: T.List[str] = []
    for i in args:
        if i.startswith('-D'):
            i = '-define=' + i[2:]
        if i.startswith('-I'):
            i = '-include=' + i[2:]
        if i.startswith('-Wl,-rpath='):
            continue
        elif i == '--print-search-dirs':
            continue
        elif i.startswith('-L'):
            continue
        elif not i.startswith('-lib=') and i.endswith(('.a', '.lib')):
            i = '-lib=' + i
        result.append(i)
    return result