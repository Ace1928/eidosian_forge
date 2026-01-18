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