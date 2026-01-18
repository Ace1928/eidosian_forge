import inspect
import llvmlite.binding as ll
import multiprocessing
import numpy as np
import os
import stat
import shutil
import subprocess
import sys
import traceback
import unittest
import warnings
from numba import njit
from numba.core import codegen
from numba.core.caching import _UserWideCacheLocator
from numba.core.errors import NumbaWarning
from numba.parfors import parfor
from numba.tests.support import (
from numba import njit
from numba import njit
from file2 import function2
from numba import njit
def test_outer_then_inner(self):
    mod = self.import_module()
    self.assertPreciseEqual(mod.outer(3, 2), 2)
    self.check_pycache(4)
    self.assertPreciseEqual(mod.outer_uncached(3, 2), 2)
    self.check_pycache(4)
    mod = self.import_module()
    f = mod.inner
    self.assertPreciseEqual(f(3, 2), 6)
    self.check_pycache(4)
    self.assertPreciseEqual(f(3.5, 2), 6.5)
    self.check_pycache(5)