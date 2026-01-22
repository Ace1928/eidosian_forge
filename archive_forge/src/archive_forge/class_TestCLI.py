import os
import subprocess
import sys
import threading
import json
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest import mock
import unittest
from numba.tests.support import TestCase, linux_only
import numba.misc.numba_sysinfo as nsi
from numba.tests.gdb_support import needs_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
from numba.misc.numba_gdbinfo import _GDBTestWrapper
class TestCLI(TestCase):

    def test_as_module_exit_code(self):
        cmdline = [sys.executable, '-m', 'numba']
        with self.assertRaises(AssertionError) as raises:
            run_cmd(cmdline)
        self.assertIn('process failed with code 1', str(raises.exception))

    def test_sysinfo_from_module(self):
        cmdline = [sys.executable, '-m', 'numba', '-s']
        o, _ = run_cmd(cmdline)
        self.assertIn('System info', o)

    def test_json_sysinfo_from_module(self):
        with TemporaryDirectory() as d:
            path = os.path.join(d, 'test_json_sysinfo.json')
            cmdline = [sys.executable, '-m', 'numba', '--sys-json', path]
            run_cmd(cmdline)
            with self.subTest(msg=f'{path} exists'):
                self.assertTrue(os.path.exists(path))
            with self.subTest(msg='json load'):
                with open(path, 'r') as f:
                    info = json.load(f)
            safe_contents = {int: (nsi._cpu_count,), float: (nsi._runtime,), str: (nsi._start, nsi._start_utc, nsi._machine, nsi._cpu_name, nsi._platform_name, nsi._os_name, nsi._os_version, nsi._python_comp, nsi._python_impl, nsi._python_version, nsi._llvm_version), bool: (nsi._cu_dev_init, nsi._svml_state, nsi._svml_loaded, nsi._svml_operational, nsi._llvm_svml_patched, nsi._tbb_thread, nsi._openmp_thread, nsi._wkq_thread), list: (nsi._errors, nsi._warnings), dict: (nsi._numba_env_vars,)}
            for t, keys in safe_contents.items():
                for k in keys:
                    with self.subTest(k=k):
                        self.assertIsInstance(info[k], t)

    @needs_gdb
    def test_gdb_status_from_module(self):
        cmdline = [sys.executable, '-m', 'numba', '-g']
        o, _ = run_cmd(cmdline)
        self.assertIn('GDB info', o)
        self.assertIn('Numba printing extension support', o)