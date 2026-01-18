import os
import subprocess
import sys
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core.errors import (
from numba.core import errors
from numba.tests.support import ignore_internal_warnings
def test_warnings_fixer(self):
    wfix = errors.WarningsFixer(errors.NumbaWarning)
    with wfix.catch_warnings('foo', 10):
        warnings.warn(errors.NumbaWarning('same'))
        warnings.warn(errors.NumbaDeprecationWarning('same'))
        ignore_internal_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ignore_internal_warnings()
        wfix.flush()
        self.assertEqual(len(w), 2)
        self.assertEqual(w[0].category, NumbaDeprecationWarning)
        self.assertEqual(w[1].category, NumbaWarning)
        self.assertIn('same', str(w[0].message))
        self.assertIn('same', str(w[1].message))