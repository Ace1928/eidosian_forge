import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
class TestMerger(TestCaseWithTransport):

    def set_up_trees(self):
        this = self.make_branch_and_tree('this')
        this.commit('rev1', rev_id=b'rev1')
        other = this.controldir.sprout('other').open_workingtree()
        this.commit('rev2a', rev_id=b'rev2a')
        other.commit('rev2b', rev_id=b'rev2b')
        return (this, other)

    def test_from_revision_ids(self):
        this, other = self.set_up_trees()
        self.assertRaises(errors.NoSuchRevision, Merger.from_revision_ids, this, b'rev2b')
        this.lock_write()
        self.addCleanup(this.unlock)
        merger = Merger.from_revision_ids(this, b'rev2b', other_branch=other.branch)
        self.assertEqual(b'rev2b', merger.other_rev_id)
        self.assertEqual(b'rev1', merger.base_rev_id)
        merger = Merger.from_revision_ids(this, b'rev2b', b'rev2a', other_branch=other.branch)
        self.assertEqual(b'rev2a', merger.base_rev_id)

    def test_from_uncommitted(self):
        this, other = self.set_up_trees()
        merger = Merger.from_uncommitted(this, other, None)
        self.assertIs(other, merger.other_tree)
        self.assertIs(None, merger.other_rev_id)
        self.assertEqual(b'rev2b', merger.base_rev_id)

    def prepare_for_merging(self):
        this, other = self.set_up_trees()
        other.commit('rev3', rev_id=b'rev3')
        this.lock_write()
        self.addCleanup(this.unlock)
        return (this, other)

    def test_from_mergeable(self):
        this, other = self.prepare_for_merging()
        md = merge_directive.MergeDirective2.from_objects(other.branch.repository, b'rev3', 0, 0, 'this')
        other.lock_read()
        self.addCleanup(other.unlock)
        merger, verified = Merger.from_mergeable(this, md)
        md.patch = None
        merger, verified = Merger.from_mergeable(this, md)
        self.assertEqual('inapplicable', verified)
        self.assertEqual(b'rev3', merger.other_rev_id)
        self.assertEqual(b'rev1', merger.base_rev_id)
        md.base_revision_id = b'rev2b'
        merger, verified = Merger.from_mergeable(this, md)
        self.assertEqual(b'rev2b', merger.base_rev_id)

    def test_from_mergeable_old_merge_directive(self):
        this, other = self.prepare_for_merging()
        other.lock_write()
        self.addCleanup(other.unlock)
        md = merge_directive.MergeDirective.from_objects(other.branch.repository, b'rev3', 0, 0, 'this')
        merger, verified = Merger.from_mergeable(this, md)
        self.assertEqual(b'rev3', merger.other_rev_id)
        self.assertEqual(b'rev1', merger.base_rev_id)