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
def test_run_brz_subprocess(self):
    """The run_bzr_helper_external command behaves nicely."""
    self.assertRunBzrSubprocess({'process_args': ['--version']}, StubProcess(), '--version')
    self.assertRunBzrSubprocess({'process_args': ['--version']}, StubProcess(), ['--version'])
    result = self.assertRunBzrSubprocess({}, StubProcess(retcode=3), '--version', retcode=None)
    result = self.assertRunBzrSubprocess({}, StubProcess(out='is free software'), '--version')
    self.assertContainsRe(result[0], 'is free software')
    self.assertRaises(AssertionError, self.assertRunBzrSubprocess, {'process_args': ['--versionn']}, StubProcess(retcode=3), '--versionn')
    result = self.assertRunBzrSubprocess({}, StubProcess(retcode=3), '--versionn', retcode=3)
    result = self.assertRunBzrSubprocess({}, StubProcess(err='unknown command', retcode=3), '--versionn', retcode=None)
    self.assertContainsRe(result[1], 'unknown command')