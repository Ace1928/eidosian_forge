import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def _captured_expect(self, expect):
    try:
        self._captured.expect(expect, timeout=self._timeout)
    except pexpect.exceptions.TIMEOUT as e:
        msg = f'Expected value did not arrive: {expect}.'
        raise ValueError(msg) from e