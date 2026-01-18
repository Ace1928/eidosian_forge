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
def test_profiles_tests(self):
    self.requireFeature(features.lsprof_feature)
    terminal = testtools.testresult.doubles.ExtendedTestResult()
    result = tests.ProfileResult(terminal)

    class Sample(tests.TestCase):

        def a(self):
            self.sample_function()

        def sample_function(self):
            pass
    test = Sample('a')
    test.run(result)
    case = terminal._events[0][1]
    self.assertLength(1, case._benchcalls)
    (_, _, _), stats = case._benchcalls[0]
    self.assertTrue(callable(stats.pprint))