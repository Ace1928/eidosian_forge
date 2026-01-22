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
class TestGDBCLIInfo(TestCase):

    def setUp(self):
        self._patches = []
        mock_init = lambda self: None
        self._patches.append(mock.patch.object(_GDBTestWrapper, '__init__', mock_init))
        bpath = 'numba.misc.numba_gdbinfo._GDBTestWrapper.gdb_binary'
        self._patches.append(mock.patch(bpath, 'PATH_TO_GDB'))

        def _patch(fnstr, func):
            self._patches.append(mock.patch.object(_GDBTestWrapper, fnstr, func))

        def mock_check_launch(self):
            return CompletedProcess('COMMAND STRING', 0)
        _patch('check_launch', mock_check_launch)

        def mock_check_python(self):
            return CompletedProcess('COMMAND STRING', 0, stdout='(3, 2)', stderr='')
        _patch('check_python', mock_check_python)

        def mock_check_numpy(self):
            return CompletedProcess('COMMAND STRING', 0, stdout='True', stderr='')
        _patch('check_numpy', mock_check_numpy)

        def mock_check_numpy_version(self):
            return CompletedProcess('COMMAND STRING', 0, stdout='1.15', stderr='')
        _patch('check_numpy_version', mock_check_numpy_version)
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_valid(self):
        collected = collect_gdbinfo()
        self.assertEqual(collected.binary_loc, 'PATH_TO_GDB')
        extp = os.path.exists(os.path.abspath(collected.extension_loc))
        self.assertTrue(extp)
        self.assertEqual(collected.py_ver, '3.2')
        self.assertEqual(collected.np_ver, '1.15')
        self.assertIn('Full', collected.supported)

    def test_invalid_binary(self):

        def mock_fn(self):
            return CompletedProcess('INVALID_BINARY', 1)
        with mock.patch.object(_GDBTestWrapper, 'check_launch', mock_fn):
            info = collect_gdbinfo()
            self.assertIn('Testing gdb binary failed.', info.binary_loc)
            self.assertIn("gdb at 'PATH_TO_GDB' does not appear to work", info.binary_loc)

    def test_no_python(self):

        def mock_fn(self):
            return CompletedProcess('NO PYTHON', 1)
        with mock.patch.object(_GDBTestWrapper, 'check_python', mock_fn):
            collected = collect_gdbinfo()
            self.assertEqual(collected.py_ver, 'No Python support')
            self.assertEqual(collected.supported, 'None')

    def test_unparsable_python_version(self):

        def mock_fn(self):
            return CompletedProcess('NO PYTHON', 0, stdout='(NOT A VERSION)')
        with mock.patch.object(_GDBTestWrapper, 'check_python', mock_fn):
            collected = collect_gdbinfo()
            self.assertEqual(collected.py_ver, 'No Python support')

    def test_no_numpy(self):

        def mock_fn(self):
            return CompletedProcess('NO NUMPY', 1)
        with mock.patch.object(_GDBTestWrapper, 'check_numpy', mock_fn):
            collected = collect_gdbinfo()
            self.assertEqual(collected.np_ver, 'No NumPy support')
            self.assertEqual(collected.py_ver, '3.2')
            self.assertIn('Partial', collected.supported)

    def test_no_numpy_version(self):

        def mock_fn(self):
            return CompletedProcess('NO NUMPY VERSION', 1)
        with mock.patch.object(_GDBTestWrapper, 'check_numpy_version', mock_fn):
            collected = collect_gdbinfo()
            self.assertEqual(collected.np_ver, 'Unknown')

    def test_traceback_in_numpy_version(self):

        def mock_fn(self):
            return CompletedProcess('NO NUMPY VERSION', 0, stdout='(NOT A VERSION)', stderr='Traceback')
        with mock.patch.object(_GDBTestWrapper, 'check_numpy_version', mock_fn):
            collected = collect_gdbinfo()
            self.assertEqual(collected.np_ver, 'Unknown')