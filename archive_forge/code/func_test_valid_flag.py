import unittest
import warnings
from contextlib import contextmanager
import numpy as np
import llvmlite.binding as llvm
from numba import njit, types
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.codegen import _parse_refprune_flags
from numba.tests.support import override_config, TestCase
def test_valid_flag(self):
    with set_refprune_flags('per_bb, diamond, fanout,fanout_raise'):
        optval = _parse_refprune_flags()
        self.assertEqual(optval, llvm.RefPruneSubpasses.ALL)