import os
import platform
import re
import subprocess
import sys
import threading
from itertools import permutations
from numba import njit, gdb, gdb_init, gdb_breakpoint, prange
from numba.core import errors
from numba import jit
from numba.tests.support import (TestCase, captured_stdout, tag,
from numba.tests.gdb_support import needs_gdb
import unittest
@not_arm
@unix_only
@needs_gdb
class TestGdbBinding(TestCase):
    """
    This test class is used to generate tests which will run the test cases
    defined in TestGdbBindImpls in isolated subprocesses, this is for safety
    in case something goes awry.
    """
    _numba_parallel_test_ = False
    _DEBUG = True

    def run_cmd(self, cmdline, env, kill_is_ok=False):
        popen = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, shell=True)

        def kill():
            popen.stdout.flush()
            popen.stderr.flush()
            popen.kill()
        timeout = threading.Timer(20.0, kill)
        try:
            timeout.start()
            out, err = popen.communicate()
            retcode = popen.returncode
            if retcode != 0:
                raise AssertionError('process failed with code %s: stderr follows\n%s\nstdout :%s' % (retcode, err.decode(), out.decode()))
            return (out.decode(), err.decode())
        finally:
            timeout.cancel()
        return (None, None)

    def run_test_in_separate_process(self, test, **kwargs):
        env_copy = os.environ.copy()
        env_copy['NUMBA_OPT'] = '1'
        env_copy['GDB_TEST'] = '1'
        cmdline = [sys.executable, '-m', 'numba.runtests', test]
        return self.run_cmd(' '.join(cmdline), env_copy, **kwargs)

    @classmethod
    def _inject(cls, name):
        themod = TestGdbBindImpls.__module__
        thecls = TestGdbBindImpls.__name__
        assert name.endswith('_impl')
        methname = name.replace('_impl', '')
        injected_method = '%s.%s.%s' % (themod, thecls, name)

        def test_template(self):
            o, e = self.run_test_in_separate_process(injected_method)
            dbgmsg = f'\nSTDOUT={o}\nSTDERR={e}\n'
            m = re.search("\\.\\.\\. skipped '(.*?)'", e)
            if m is not None:
                self.skipTest(m.group(1))
            self.assertIn('GNU gdb', o, msg=dbgmsg)
            self.assertIn('OK', e, msg=dbgmsg)
            self.assertNotIn('FAIL', e, msg=dbgmsg)
            self.assertNotIn('ERROR', e, msg=dbgmsg)
        if 'quick' in name:
            setattr(cls, methname, test_template)
        else:
            setattr(cls, methname, long_running(test_template))

    @classmethod
    def generate(cls):
        for name in dir(TestGdbBindImpls):
            if name.startswith('test_gdb'):
                cls._inject(name)