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
class TestWorkingTreeScenarios(tests.TestCase):

    def test_scenarios(self):
        from .per_workingtree import make_scenarios
        server1 = 'a'
        server2 = 'b'
        formats = [workingtree_4.WorkingTreeFormat4(), workingtree_3.WorkingTreeFormat3(), workingtree_4.WorkingTreeFormat6()]
        scenarios = make_scenarios(server1, server2, formats, remote_server='c', remote_readonly_server='d', remote_backing_server='e')
        self.assertEqual([('WorkingTreeFormat4', {'bzrdir_format': formats[0]._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': formats[0]}), ('WorkingTreeFormat3', {'bzrdir_format': formats[1]._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': formats[1]}), ('WorkingTreeFormat6', {'bzrdir_format': formats[2]._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': formats[2]}), ('WorkingTreeFormat6,remote', {'bzrdir_format': formats[2]._matchingcontroldir, 'repo_is_remote': True, 'transport_readonly_server': 'd', 'transport_server': 'c', 'vfs_transport_factory': 'e', 'workingtree_format': formats[2]})], scenarios)