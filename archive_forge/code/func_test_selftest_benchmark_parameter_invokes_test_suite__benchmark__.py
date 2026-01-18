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
def test_selftest_benchmark_parameter_invokes_test_suite__benchmark__(self):
    factory_called = []

    def factory():
        factory_called.append(True)
        return TestUtil.TestSuite()
    out = StringIO()
    err = StringIO()
    self.apply_redirected(out, err, None, breezy.tests.selftest, test_suite_factory=factory)
    self.assertEqual([True], factory_called)