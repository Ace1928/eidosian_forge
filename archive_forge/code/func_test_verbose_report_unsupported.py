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
def test_verbose_report_unsupported(self):
    result_stream = StringIO()
    result = breezy.tests.VerboseTestResult(result_stream, descriptions=0, verbosity=2)
    test = self.get_passing_test()
    feature = features.Feature()
    result.startTest(test)
    prefix = len(result_stream.getvalue())
    result.report_unsupported(test, feature)
    output = result_stream.getvalue()[prefix:]
    lines = output.splitlines()
    self.assertStartsWith(lines[0], 'NODEP')
    self.assertEqual(lines[1], "    The feature 'Feature' is not available.")