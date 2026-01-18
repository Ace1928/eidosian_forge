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
def test_overrideAttr_without_value(self):
    self.test_attr = 'original'
    obj = self

    class Test(tests.TestCase):

        def setUp(self):
            super().setUp()
            self.orig = self.overrideAttr(obj, 'test_attr')

        def test_value(self):
            self.assertEqual('original', self.orig)
            self.assertEqual('original', obj.test_attr)
            obj.test_attr = 'modified'
            self.assertEqual('modified', obj.test_attr)
    self._run_successful_test(Test('test_value'))
    self.assertEqual('original', obj.test_attr)