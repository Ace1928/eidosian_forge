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
class TestInterRepositoryScenarios(tests.TestCase):

    def test_scenarios(self):
        from .per_interrepository import make_scenarios
        server1 = 'a'
        server2 = 'b'
        formats = [('C0', 'C1', 'C2', 'C3'), ('D0', 'D1', 'D2', 'D3')]
        scenarios = make_scenarios(server1, server2, formats)
        self.assertEqual([('C0,str,str', {'repository_format': 'C1', 'repository_format_to': 'C2', 'transport_readonly_server': 'b', 'transport_server': 'a', 'extra_setup': 'C3'}), ('D0,str,str', {'repository_format': 'D1', 'repository_format_to': 'D2', 'transport_readonly_server': 'b', 'transport_server': 'a', 'extra_setup': 'D3'})], scenarios)