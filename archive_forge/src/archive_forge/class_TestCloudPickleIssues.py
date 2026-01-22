import contextlib
import gc
import pickle
import runpy
import subprocess
import sys
import unittest
from multiprocessing import get_context
import numba
from numba.core.errors import TypingError
from numba.tests.support import TestCase
from numba.core.target_extension import resolve_dispatcher_from_str
from numba.cloudpickle import dumps, loads
class TestCloudPickleIssues(TestCase):
    """This test case includes issues specific to the cloudpickle implementation.
    """
    _numba_parallel_test_ = False

    def test_dynamic_class_reset_on_unpickle(self):

        class Klass:
            classvar = None

        def mutator():
            Klass.classvar = 100

        def check():
            self.assertEqual(Klass.classvar, 100)
        saved = dumps(Klass)
        mutator()
        check()
        loads(saved)
        check()
        loads(saved)
        check()

    @unittest.skipIf(__name__ == '__main__', 'Test cannot run as when module is __main__')
    def test_main_class_reset_on_unpickle(self):
        mp = get_context('spawn')
        proc = mp.Process(target=check_main_class_reset_on_unpickle)
        proc.start()
        proc.join(timeout=60)
        self.assertEqual(proc.exitcode, 0)

    def test_dynamic_class_reset_on_unpickle_new_proc(self):

        class Klass:
            classvar = None
        saved = dumps(Klass)
        mp = get_context('spawn')
        proc = mp.Process(target=check_unpickle_dyn_class_new_proc, args=(saved,))
        proc.start()
        proc.join(timeout=60)
        self.assertEqual(proc.exitcode, 0)

    def test_dynamic_class_issue_7356(self):
        cfunc = numba.njit(issue_7356)
        self.assertEqual(cfunc(), (100, 100))