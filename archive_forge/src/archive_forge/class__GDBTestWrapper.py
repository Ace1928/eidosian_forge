from collections import namedtuple
import os
import re
import subprocess
from textwrap import dedent
from numba import config
class _GDBTestWrapper:
    """Wraps the gdb binary and has methods for checking what the gdb binary
    has support for (Python and NumPy)."""

    def __init__(self):
        gdb_binary = config.GDB_BINARY
        if gdb_binary is None:
            msg = f'No valid binary could be found for gdb named: {config.GDB_BINARY}'
            raise ValueError(msg)
        self._gdb_binary = gdb_binary

    def _run_cmd(self, cmd=()):
        gdb_call = [self.gdb_binary, '-q']
        for x in cmd:
            gdb_call.append('-ex')
            gdb_call.append(x)
        gdb_call.extend(['-ex', 'q'])
        return subprocess.run(gdb_call, capture_output=True, timeout=10, text=True)

    @property
    def gdb_binary(self):
        return self._gdb_binary

    @classmethod
    def success(cls, status):
        return status.returncode == 0

    def check_launch(self):
        """Checks that gdb will launch ok"""
        return self._run_cmd()

    def check_python(self):
        cmd = 'python from __future__ import print_function; import sys; print(sys.version_info[:2])'
        return self._run_cmd((cmd,))

    def check_numpy(self):
        cmd = 'python from __future__ import print_function; import types; import numpy; print(isinstance(numpy, types.ModuleType))'
        return self._run_cmd((cmd,))

    def check_numpy_version(self):
        cmd = 'python from __future__ import print_function; import types; import numpy;print(numpy.__version__)'
        return self._run_cmd((cmd,))