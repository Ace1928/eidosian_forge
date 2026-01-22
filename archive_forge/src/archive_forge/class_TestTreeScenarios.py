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
class TestTreeScenarios(tests.TestCase):

    def test_scenarios(self):
        from .per_tree import _dirstate_tree_from_workingtree, make_scenarios, preview_tree_post, preview_tree_pre, return_parameter, revision_tree_from_workingtree
        server1 = 'a'
        server2 = 'b'
        smart_server = test_server.SmartTCPServer_for_testing
        smart_readonly_server = test_server.ReadonlySmartTCPServer_for_testing
        mem_server = memory.MemoryServer
        formats = [workingtree_4.WorkingTreeFormat4(), workingtree_3.WorkingTreeFormat3()]
        scenarios = make_scenarios(server1, server2, formats)
        self.assertEqual(9, len(scenarios))
        default_wt_format = workingtree.format_registry.get_default()
        wt4_format = workingtree_4.WorkingTreeFormat4()
        wt5_format = workingtree_4.WorkingTreeFormat5()
        wt6_format = workingtree_4.WorkingTreeFormat6()
        git_wt_format = git_workingtree.GitWorkingTreeFormat()
        expected_scenarios = [('WorkingTreeFormat4', {'bzrdir_format': formats[0]._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': formats[0], '_workingtree_to_test_tree': return_parameter}), ('WorkingTreeFormat3', {'bzrdir_format': formats[1]._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': formats[1], '_workingtree_to_test_tree': return_parameter}), ('WorkingTreeFormat6,remote', {'bzrdir_format': wt6_format._matchingcontroldir, 'repo_is_remote': True, 'transport_readonly_server': smart_readonly_server, 'transport_server': smart_server, 'vfs_transport_factory': mem_server, 'workingtree_format': wt6_format, '_workingtree_to_test_tree': return_parameter}), ('RevisionTree', {'_workingtree_to_test_tree': revision_tree_from_workingtree, 'bzrdir_format': default_wt_format._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': default_wt_format}), ('GitRevisionTree', {'_workingtree_to_test_tree': revision_tree_from_workingtree, 'bzrdir_format': git_wt_format._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': git_wt_format}), ('DirStateRevisionTree,WT4', {'_workingtree_to_test_tree': _dirstate_tree_from_workingtree, 'bzrdir_format': wt4_format._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': wt4_format}), ('DirStateRevisionTree,WT5', {'_workingtree_to_test_tree': _dirstate_tree_from_workingtree, 'bzrdir_format': wt5_format._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': wt5_format}), ('PreviewTree', {'_workingtree_to_test_tree': preview_tree_pre, 'bzrdir_format': default_wt_format._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': default_wt_format}), ('PreviewTreePost', {'_workingtree_to_test_tree': preview_tree_post, 'bzrdir_format': default_wt_format._matchingcontroldir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': default_wt_format})]
        self.assertEqual(expected_scenarios, scenarios)