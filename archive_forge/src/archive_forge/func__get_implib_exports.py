from __future__ import annotations
import typing as T
import os, sys
from .. import mesonlib
from .. import mlog
from ..mesonlib import Popen_safe
import argparse
def _get_implib_exports(impfilename: str) -> T.Tuple[T.List[str], str]:
    all_stderr = ''
    env = os.environ.copy()
    env['VSLANG'] = '1033'
    output, e = call_tool_nowarn(get_tool('dumpbin') + ['-exports', impfilename], env=env)
    if output:
        lines = output.split('\n')
        start = lines.index('File Type: LIBRARY')
        end = lines.index('  Summary')
        return (lines[start:end], None)
    all_stderr += e
    for nm in ('llvm-nm', 'nm'):
        output, e = call_tool_nowarn(get_tool(nm) + ['--extern-only', '--defined-only', '--format=posix', impfilename])
        if output:
            result = []
            for line in output.split('\n'):
                if ' T ' not in line or line.startswith('.text'):
                    continue
                result.append(line.split(maxsplit=1)[0])
            return (result, None)
        all_stderr += e
    return ([], all_stderr)