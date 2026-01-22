from __future__ import annotations
from .common import CMakeException
from .generator import parse_generator_expressions
from .. import mlog
from ..mesonlib import version_compare
import typing as T
from pathlib import Path
from functools import lru_cache
import re
import json
import textwrap
def _cmake_set_target_properties(self, tline: CMakeTraceLine) -> None:
    args = list(tline.args)
    targets = []
    while args:
        curr = args.pop(0)
        if curr == 'PROPERTIES':
            break
        targets.append(curr)
    arglist: T.List[T.Tuple[str, T.List[str]]] = []
    if self.trace_format == 'human':
        name = args.pop(0)
        values: T.List[str] = []
        prop_regex = re.compile('^[A-Z_]+$')
        for a in args:
            if prop_regex.match(a):
                if values:
                    arglist.append((name, ' '.join(values).split(';')))
                name = a
                values = []
            else:
                values.append(a)
        if values:
            arglist.append((name, ' '.join(values).split(';')))
    else:
        arglist = [(x[0], x[1].split(';')) for x in zip(args[::2], args[1::2])]
    for name, value in arglist:
        for i in targets:
            if i not in self.targets:
                return self._gen_exception('set_target_properties', f'TARGET {i} not found', tline)
            self.targets[i].properties[name] = value