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
def test_predefined_prefixes(self):
    tpr = tests.test_prefix_alias_registry
    self.assertEqual('breezy', tpr.resolve_alias('breezy'))
    self.assertEqual('breezy.doc', tpr.resolve_alias('bd'))
    self.assertEqual('breezy.utils', tpr.resolve_alias('bu'))
    self.assertEqual('breezy.tests', tpr.resolve_alias('bt'))
    self.assertEqual('breezy.tests.blackbox', tpr.resolve_alias('bb'))
    self.assertEqual('breezy.plugins', tpr.resolve_alias('bp'))