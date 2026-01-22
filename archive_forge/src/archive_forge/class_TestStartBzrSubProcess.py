import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
class TestStartBzrSubProcess(tests.TestCase):
    """Stub test start_brz_subprocess."""

    def _subprocess_log_cleanup(self):
        """Inhibits the base version as we don't produce a log file."""

    def _popen(self, *args, **kwargs):
        """Override the base version to record the command that is run.

        From there we can ensure it is correct without spawning a real process.
        """
        self.check_popen_state()
        self._popen_args = args
        self._popen_kwargs = kwargs
        raise _DontSpawnProcess()

    def check_popen_state(self):
        """Replace to make assertions when popen is called."""

    def test_run_brz_subprocess_no_plugins(self):
        self.assertRaises(_DontSpawnProcess, self.start_brz_subprocess, [])
        command = self._popen_args[0]
        if self.get_brz_path().endswith('__main__.py'):
            self.assertEqual(sys.executable, command[0])
            self.assertEqual('-m', command[1])
            self.assertEqual('breezy', command[2])
            rest = command[3:]
        else:
            self.assertEqual(self.get_brz_path(), command[0])
            rest = command[1:]
        self.assertEqual(['--no-plugins'], rest)

    def test_allow_plugins(self):
        self.assertRaises(_DontSpawnProcess, self.start_brz_subprocess, [], allow_plugins=True)
        command = self._popen_args[0]
        if self.get_brz_path().endswith('__main__.py'):
            rest = command[3:]
        else:
            rest = command[1:]
        self.assertEqual([], rest)

    def test_set_env(self):
        self.assertFalse('EXISTANT_ENV_VAR' in os.environ)

        def check_environment():
            self.assertEqual('set variable', os.environ['EXISTANT_ENV_VAR'])
        self.check_popen_state = check_environment
        self.assertRaises(_DontSpawnProcess, self.start_brz_subprocess, [], env_changes={'EXISTANT_ENV_VAR': 'set variable'})
        self.assertFalse('EXISTANT_ENV_VAR' in os.environ)

    def test_run_brz_subprocess_env_del(self):
        """run_brz_subprocess can remove environment variables too."""
        self.assertFalse('EXISTANT_ENV_VAR' in os.environ)

        def check_environment():
            self.assertFalse('EXISTANT_ENV_VAR' in os.environ)
        os.environ['EXISTANT_ENV_VAR'] = 'set variable'
        self.check_popen_state = check_environment
        self.assertRaises(_DontSpawnProcess, self.start_brz_subprocess, [], env_changes={'EXISTANT_ENV_VAR': None})
        self.assertEqual('set variable', os.environ['EXISTANT_ENV_VAR'])
        del os.environ['EXISTANT_ENV_VAR']

    def test_env_del_missing(self):
        self.assertFalse('NON_EXISTANT_ENV_VAR' in os.environ)

        def check_environment():
            self.assertFalse('NON_EXISTANT_ENV_VAR' in os.environ)
        self.check_popen_state = check_environment
        self.assertRaises(_DontSpawnProcess, self.start_brz_subprocess, [], env_changes={'NON_EXISTANT_ENV_VAR': None})

    def test_working_dir(self):
        """Test that we can specify the working dir for the child"""
        chdirs = []

        def chdir(path):
            chdirs.append(path)
        self.overrideAttr(os, 'chdir', chdir)

        def getcwd():
            return 'current'
        self.overrideAttr(osutils, 'getcwd', getcwd)
        self.assertRaises(_DontSpawnProcess, self.start_brz_subprocess, [], working_dir='foo')
        self.assertEqual(['foo', 'current'], chdirs)

    def test_get_brz_path_with_cwd_breezy(self):
        self.get_source_path = lambda: ''
        self.overrideAttr(os.path, 'isfile', lambda path: True)
        self.assertEqual(self.get_brz_path(), 'brz')