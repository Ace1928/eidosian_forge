import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
@skip_parfors_unsupported
class TestParforsRunner(TestCase):
    _numba_parallel_test_ = False
    _TIMEOUT = 1800 if platform.machine() != 'aarch64' else 3600
    'This is the test runner for all the parfors tests, it runs them in\n    subprocesses as described above. The convention for the test method naming\n    is: `test_<TestClass>` where <TestClass> is the name of the test class in\n    this module.\n    '

    def runner(self):
        themod = self.__module__
        test_clazz_name = self.id().split('.')[-1].split('_')[-1]
        self.subprocess_test_runner(test_module=themod, test_class=test_clazz_name, timeout=self._TIMEOUT)

    def test_TestParforBasic(self):
        self.runner()

    def test_TestParforNumericalMisc(self):
        self.runner()

    def test_TestParforNumPy(self):
        self.runner()

    def test_TestParfors(self):
        self.runner()

    def test_TestParforsBitMask(self):
        self.runner()

    def test_TestParforsDiagnostics(self):
        self.runner()

    def test_TestParforsLeaks(self):
        self.runner()

    def test_TestParforsMisc(self):
        self.runner()

    def test_TestParforsOptions(self):
        self.runner()

    def test_TestParforsSlice(self):
        self.runner()

    def test_TestParforsVectorizer(self):
        self.runner()

    def test_TestPrangeBasic(self):
        self.runner()

    def test_TestPrangeSpecific(self):
        self.runner()