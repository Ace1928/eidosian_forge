import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
def write_statistic(self, stat):
    if stat.supported == 0:
        self.print('This module is not supported.')
    else:
        msg = 'Not showing {} unsupported functions.'
        self.print(msg.format(stat.unsupported))
        self.print()
        self.print(stat.describe())
    self.print()