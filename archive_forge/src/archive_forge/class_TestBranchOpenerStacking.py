from typing import List
from .. import urlutils
from ..branch import Branch
from ..bzr import BzrProber
from ..bzr.branch import BranchReferenceFormat
from ..controldir import ControlDir, ControlDirFormat
from ..errors import NotBranchError, RedirectRequested
from ..transport import (Transport, chroot, get_transport, register_transport,
from ..url_policy_open import (BadUrl, BranchLoopError, BranchOpener,
from . import TestCase, TestCaseWithTransport
class TestBranchOpenerStacking(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        BranchOpener.install_hook()

    def make_branch_opener(self, allowed_urls, probers=None):
        policy = WhitelistPolicy(True, allowed_urls, True)
        return BranchOpener(policy, probers)

    def test_probers(self):
        b = self.make_branch('branch')
        opener = self.make_branch_opener([b.base], probers=[])
        self.assertRaises(NotBranchError, opener.open, b.base)
        opener = self.make_branch_opener([b.base], probers=[BzrProber])
        self.assertEqual(b.base, opener.open(b.base).base)

    def test_default_probers(self):
        self.addCleanup(ControlDirFormat.unregister_prober, TrackingProber)
        ControlDirFormat.register_prober(TrackingProber)
        TrackingProber.seen_urls = []
        opener = self.make_branch_opener(['.'], probers=[TrackingProber])
        self.assertRaises(NotBranchError, opener.open, '.')
        self.assertEqual(1, len(TrackingProber.seen_urls))
        TrackingProber.seen_urls = []
        self.assertRaises(NotBranchError, ControlDir.open, '.')
        self.assertEqual(1, len(TrackingProber.seen_urls))

    def test_allowed_url(self):
        stacked_on_branch = self.make_branch('base-branch', format='1.6')
        stacked_branch = self.make_branch('stacked-branch', format='1.6')
        stacked_branch.set_stacked_on_url(stacked_on_branch.base)
        opener = self.make_branch_opener([stacked_branch.base, stacked_on_branch.base])
        opener.open(stacked_branch.base)

    def test_nstackable_repository(self):
        branch = self.make_branch('unstacked', format='knit')
        opener = self.make_branch_opener([branch.base])
        opener.open(branch.base)

    def test_allowed_relative_url(self):
        stacked_on_branch = self.make_branch('base-branch', format='1.6')
        stacked_branch = self.make_branch('stacked-branch', format='1.6')
        stacked_branch.set_stacked_on_url('../base-branch')
        opener = self.make_branch_opener([stacked_branch.base, stacked_on_branch.base])
        self.assertNotEqual('../base-branch', stacked_on_branch.base)
        opener.open(stacked_branch.base)

    def test_allowed_relative_nested(self):
        self.get_transport().mkdir('subdir')
        a = self.make_branch('subdir/a', format='1.6')
        b = self.make_branch('b', format='1.6')
        b.set_stacked_on_url('../subdir/a')
        c = self.make_branch('subdir/c', format='1.6')
        c.set_stacked_on_url('../../b')
        opener = self.make_branch_opener([c.base, b.base, a.base])
        opener.open(c.base)

    def test_forbidden_url(self):
        stacked_on_branch = self.make_branch('base-branch', format='1.6')
        stacked_branch = self.make_branch('stacked-branch', format='1.6')
        stacked_branch.set_stacked_on_url(stacked_on_branch.base)
        opener = self.make_branch_opener([stacked_branch.base])
        self.assertRaises(BadUrl, opener.open, stacked_branch.base)

    def test_forbidden_url_nested(self):
        a = self.make_branch('a', format='1.6')
        b = self.make_branch('b', format='1.6')
        b.set_stacked_on_url(a.base)
        c = self.make_branch('c', format='1.6')
        c.set_stacked_on_url(b.base)
        opener = self.make_branch_opener([c.base, b.base])
        self.assertRaises(BadUrl, opener.open, c.base)

    def test_self_stacked_branch(self):
        a = self.make_branch('a', format='1.6')
        a.get_config().set_user_option('stacked_on_location', a.base)
        opener = self.make_branch_opener([a.base])
        self.assertRaises(BranchLoopError, opener.open, a.base)

    def test_loop_stacked_branch(self):
        a = self.make_branch('a', format='1.6')
        b = self.make_branch('b', format='1.6')
        a.set_stacked_on_url(b.base)
        b.set_stacked_on_url(a.base)
        opener = self.make_branch_opener([a.base, b.base])
        self.assertRaises(BranchLoopError, opener.open, a.base)
        self.assertRaises(BranchLoopError, opener.open, b.base)

    def test_custom_opener(self):
        a = self.make_branch('a', format='2a')
        b = self.make_branch('b', format='2a')
        b.set_stacked_on_url(a.base)
        TrackingProber.seen_urls = []
        opener = self.make_branch_opener([a.base, b.base], probers=[TrackingProber])
        opener.open(b.base)
        self.assertEqual(set(TrackingProber.seen_urls), {b.base, a.base})

    def test_custom_opener_with_branch_reference(self):
        a = self.make_branch('a', format='2a')
        b_dir = self.make_controldir('b')
        b = BranchReferenceFormat().initialize(b_dir, target_branch=a)
        TrackingProber.seen_urls = []
        opener = self.make_branch_opener([a.base, b.base], probers=[TrackingProber])
        opener.open(b.base)
        self.assertEqual(set(TrackingProber.seen_urls), {b.base, a.base})