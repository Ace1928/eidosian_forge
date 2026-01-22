import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
class TestBzrDirFormat(TestCaseWithTransport):
    """Tests for the BzrDirFormat facility."""

    def test_find_format(self):
        bzr.BzrProber.formats.register(BzrDirFormatTest1.get_format_string(), BzrDirFormatTest1())
        self.addCleanup(bzr.BzrProber.formats.remove, BzrDirFormatTest1.get_format_string())
        bzr.BzrProber.formats.register(BzrDirFormatTest2.get_format_string(), BzrDirFormatTest2())
        self.addCleanup(bzr.BzrProber.formats.remove, BzrDirFormatTest2.get_format_string())
        t = self.get_transport()
        self.build_tree(['foo/', 'bar/'], transport=t)

        def check_format(format, url):
            format.initialize(url)
            t = _mod_transport.get_transport_from_path(url)
            found_format = bzrdir.BzrDirFormat.find_format(t)
            self.assertIsInstance(found_format, format.__class__)
        check_format(BzrDirFormatTest1(), 'foo')
        check_format(BzrDirFormatTest2(), 'bar')

    def test_find_format_nothing_there(self):
        self.assertRaises(NotBranchError, bzrdir.BzrDirFormat.find_format, _mod_transport.get_transport_from_path('.'))

    def test_find_format_unknown_format(self):
        t = self.get_transport()
        t.mkdir('.bzr')
        t.put_bytes('.bzr/branch-format', b'')
        self.assertRaises(UnknownFormatError, bzrdir.BzrDirFormat.find_format, _mod_transport.get_transport_from_path('.'))

    def test_find_format_line_endings(self):
        t = self.get_transport()
        t.mkdir('.bzr')
        t.put_bytes('.bzr/branch-format', b'Corrupt line endings\r\n')
        self.assertRaises(bzr.LineEndingError, bzrdir.BzrDirFormat.find_format, _mod_transport.get_transport_from_path('.'))

    def test_find_format_html(self):
        t = self.get_transport()
        t.mkdir('.bzr')
        t.put_bytes('.bzr/branch-format', b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">')
        e = self.assertRaises(NotBranchError, bzrdir.BzrDirFormat.find_format, _mod_transport.get_transport_from_path('.'))

    def test_register_unregister_format(self):
        format = SampleBzrDirFormat()
        url = self.get_url()
        format.initialize(url)
        bzr.BzrProber.formats.register(format.get_format_string(), format)
        self.assertRaises(UnsupportedFormatError, bzrdir.BzrDir.open, url)
        self.assertRaises(UnsupportedFormatError, bzrdir.BzrDir.open_containing, url)
        t = _mod_transport.get_transport_from_url(url)
        self.assertEqual(format.open(t), bzrdir.BzrDir.open_unsupported(url))
        bzr.BzrProber.formats.remove(format.get_format_string())
        self.assertRaises(UnknownFormatError, bzrdir.BzrDir.open_unsupported, url)

    def test_create_branch_and_repo_uses_default(self):
        format = SampleBzrDirFormat()
        branch = bzrdir.BzrDir.create_branch_and_repo(self.get_url(), format=format)
        self.assertTrue(isinstance(branch, SampleBranch))

    def test_create_branch_and_repo_under_shared(self):
        format = controldir.format_registry.make_controldir('knit')
        self.make_repository('.', shared=True, format=format)
        branch = bzrdir.BzrDir.create_branch_and_repo(self.get_url('child'), format=format)
        self.assertRaises(errors.NoRepositoryPresent, branch.controldir.open_repository)

    def test_create_branch_and_repo_under_shared_force_new(self):
        format = controldir.format_registry.make_controldir('knit')
        self.make_repository('.', shared=True, format=format)
        branch = bzrdir.BzrDir.create_branch_and_repo(self.get_url('child'), force_new_repo=True, format=format)
        branch.controldir.open_repository()

    def test_create_standalone_working_tree(self):
        format = SampleBzrDirFormat()
        self.assertRaises(errors.NotLocalUrl, bzrdir.BzrDir.create_standalone_workingtree, self.get_readonly_url(), format=format)
        tree = bzrdir.BzrDir.create_standalone_workingtree('.', format=format)
        self.assertEqual('A tree', tree)

    def test_create_standalone_working_tree_under_shared_repo(self):
        format = controldir.format_registry.make_controldir('knit')
        self.make_repository('.', shared=True, format=format)
        self.assertRaises(errors.NotLocalUrl, bzrdir.BzrDir.create_standalone_workingtree, self.get_readonly_url('child'), format=format)
        tree = bzrdir.BzrDir.create_standalone_workingtree('child', format=format)
        tree.controldir.open_repository()

    def test_create_branch_convenience(self):
        format = controldir.format_registry.make_controldir('knit')
        branch = bzrdir.BzrDir.create_branch_convenience('.', format=format)
        branch.controldir.open_workingtree()
        branch.controldir.open_repository()

    def test_create_branch_convenience_possible_transports(self):
        """Check that the optional 'possible_transports' is recognized"""
        format = controldir.format_registry.make_controldir('knit')
        t = self.get_transport()
        branch = bzrdir.BzrDir.create_branch_convenience('.', format=format, possible_transports=[t])
        branch.controldir.open_workingtree()
        branch.controldir.open_repository()

    def test_create_branch_convenience_root(self):
        """Creating a branch at the root of a fs should work."""
        self.vfs_transport_factory = memory.MemoryServer
        format = controldir.format_registry.make_controldir('knit')
        branch = bzrdir.BzrDir.create_branch_convenience(self.get_url(), format=format)
        self.assertRaises(errors.NoWorkingTree, branch.controldir.open_workingtree)
        branch.controldir.open_repository()

    def test_create_branch_convenience_under_shared_repo(self):
        format = controldir.format_registry.make_controldir('knit')
        self.make_repository('.', shared=True, format=format)
        branch = bzrdir.BzrDir.create_branch_convenience('child', format=format)
        branch.controldir.open_workingtree()
        self.assertRaises(errors.NoRepositoryPresent, branch.controldir.open_repository)

    def test_create_branch_convenience_under_shared_repo_force_no_tree(self):
        format = controldir.format_registry.make_controldir('knit')
        self.make_repository('.', shared=True, format=format)
        branch = bzrdir.BzrDir.create_branch_convenience('child', force_new_tree=False, format=format)
        self.assertRaises(errors.NoWorkingTree, branch.controldir.open_workingtree)
        self.assertRaises(errors.NoRepositoryPresent, branch.controldir.open_repository)

    def test_create_branch_convenience_under_shared_repo_no_tree_policy(self):
        format = controldir.format_registry.make_controldir('knit')
        repo = self.make_repository('.', shared=True, format=format)
        repo.set_make_working_trees(False)
        branch = bzrdir.BzrDir.create_branch_convenience('child', format=format)
        self.assertRaises(errors.NoWorkingTree, branch.controldir.open_workingtree)
        self.assertRaises(errors.NoRepositoryPresent, branch.controldir.open_repository)

    def test_create_branch_convenience_under_shared_repo_no_tree_policy_force_tree(self):
        format = controldir.format_registry.make_controldir('knit')
        repo = self.make_repository('.', shared=True, format=format)
        repo.set_make_working_trees(False)
        branch = bzrdir.BzrDir.create_branch_convenience('child', force_new_tree=True, format=format)
        branch.controldir.open_workingtree()
        self.assertRaises(errors.NoRepositoryPresent, branch.controldir.open_repository)

    def test_create_branch_convenience_under_shared_repo_force_new_repo(self):
        format = controldir.format_registry.make_controldir('knit')
        self.make_repository('.', shared=True, format=format)
        branch = bzrdir.BzrDir.create_branch_convenience('child', force_new_repo=True, format=format)
        branch.controldir.open_repository()
        branch.controldir.open_workingtree()