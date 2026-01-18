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
def test_scenarios(self):
    from ..bzr.workingtree_3 import WorkingTreeFormat3
    from ..bzr.workingtree_4 import WorkingTreeFormat4
    from .per_intertree import make_scenarios
    from .per_tree import return_parameter
    input_test = TestInterTreeScenarios('test_scenarios')
    server1 = 'a'
    server2 = 'b'
    format1 = WorkingTreeFormat4()
    format2 = WorkingTreeFormat3()
    formats = [('1', str, format1, format2, 'converter1'), ('2', int, format2, format1, 'converter2')]
    scenarios = make_scenarios(server1, server2, formats)
    self.assertEqual(2, len(scenarios))
    expected_scenarios = [('1', {'bzrdir_format': format1._matchingcontroldir, 'intertree_class': formats[0][1], 'workingtree_format': formats[0][2], 'workingtree_format_to': formats[0][3], 'mutable_trees_to_test_trees': formats[0][4], '_workingtree_to_test_tree': return_parameter, 'transport_server': server1, 'transport_readonly_server': server2}), ('2', {'bzrdir_format': format2._matchingcontroldir, 'intertree_class': formats[1][1], 'workingtree_format': formats[1][2], 'workingtree_format_to': formats[1][3], 'mutable_trees_to_test_trees': formats[1][4], '_workingtree_to_test_tree': return_parameter, 'transport_server': server1, 'transport_readonly_server': server2})]
    self.assertEqual(scenarios, expected_scenarios)