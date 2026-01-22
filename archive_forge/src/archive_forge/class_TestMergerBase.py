import contextlib
import os
from .. import branch as _mod_branch
from .. import conflicts, errors, memorytree
from .. import merge as _mod_merge
from .. import option
from .. import revision as _mod_revision
from .. import tests, transform
from ..bzr import inventory, knit, versionedfile
from ..bzr.conflicts import (ContentsConflict, DeletingParent, MissingParent,
from ..conflicts import ConflictList
from ..errors import NoCommits, UnrelatedBranches
from ..merge import _PlanMerge, merge_inner, transform_tree
from ..osutils import basename, file_kind, pathjoin
from ..workingtree import PointlessMerge, WorkingTree
from . import (TestCaseWithMemoryTransport, TestCaseWithTransport, features,
class TestMergerBase(TestCaseWithMemoryTransport):
    """Common functionality for Merger tests that don't write to disk."""

    def get_builder(self):
        builder = self.make_branch_builder('path')
        builder.start_series()
        self.addCleanup(builder.finish_series)
        return builder

    def setup_simple_graph(self):
        """Create a simple 3-node graph.

        :return: A BranchBuilder
        """
        builder = self.get_builder()
        builder.build_snapshot(None, [('add', ('', None, 'directory', None))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        return builder

    def setup_criss_cross_graph(self):
        """Create a 5-node graph with a criss-cross.

        :return: A BranchBuilder
        """
        builder = self.setup_simple_graph()
        builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
        builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
        return builder

    def make_Merger(self, builder, other_revision_id, interesting_files=None):
        """Make a Merger object from a branch builder"""
        mem_tree = memorytree.MemoryTree.create_on_branch(builder.get_branch())
        mem_tree.lock_write()
        self.addCleanup(mem_tree.unlock)
        merger = _mod_merge.Merger.from_revision_ids(mem_tree, other_revision_id)
        merger.set_interesting_files(interesting_files)
        merger.merge_type = _mod_merge.Merge3Merger
        return merger