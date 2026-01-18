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
def test_omitted(self):
    cache_dir = temp_directory(self.__class__.__name__)
    ctx = multiprocessing.get_context()
    result_queue = ctx.Queue()
    proc = ctx.Process(target=omitted_child_test_wrapper, args=(result_queue, cache_dir, False))
    proc.start()
    proc.join()
    success, output = result_queue.get()
    if not success:
        self.fail(output)
    self.assertEqual(output, 1000, 'Omitted function returned an incorrect output')
    proc = ctx.Process(target=omitted_child_test_wrapper, args=(result_queue, cache_dir, True))
    proc.start()
    proc.join()
    success, output = result_queue.get()
    if not success:
        self.fail(output)
    self.assertEqual(output, 1000, 'Omitted function returned an incorrect output')