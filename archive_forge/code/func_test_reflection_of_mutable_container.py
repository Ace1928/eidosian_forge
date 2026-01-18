import warnings
import unittest
from contextlib import contextmanager
from numba import jit, vectorize, guvectorize
from numba.core.errors import (NumbaDeprecationWarning,
from numba.tests.support import TestCase, needs_setuptools
@TestCase.run_test_in_subprocess
def test_reflection_of_mutable_container(self):

    def foo_list(a):
        return a.append(1)

    def foo_set(a):
        return a.add(1)
    for f in [foo_list, foo_set]:
        container = f.__name__.strip('foo_')
        inp = eval(container)([10])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore', category=NumbaWarning)
            warnings.simplefilter('always', category=NumbaPendingDeprecationWarning)
            jit(nopython=True)(f)(inp)
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, NumbaPendingDeprecationWarning)
            warn_msg = str(w[0].message)
            msg = 'Encountered the use of a type that is scheduled for deprecation'
            self.assertIn(msg, warn_msg)
            msg = "'reflected %s' found for argument" % container
            self.assertIn(msg, warn_msg)
            self.assertIn('https://numba.readthedocs.io', warn_msg)