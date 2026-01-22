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
class TestMergeInto(TestMergeIntoBase):

    def test_newdir_with_unique_roots(self):
        """Merge a branch with a unique root into a new directory."""
        project_wt, lib_wt = self.setup_two_branches()
        self.do_merge_into('lib1', 'project/lib1')
        project_wt.lock_read()
        self.addCleanup(project_wt.unlock)
        self.assertEqual([b'r1-project', b'r1-lib1'], project_wt.get_parent_ids())
        self.assertTreeEntriesEqual([('', b'project-root-id'), ('README', b'project-README-id'), ('dir', b'project-dir-id'), ('lib1', b'lib1-root-id'), ('dir/file.c', b'project-file.c-id'), ('lib1/Makefile', b'lib1-Makefile-id'), ('lib1/README', b'lib1-README-id'), ('lib1/foo.c', b'lib1-foo.c-id')], project_wt)

    def test_subdir(self):
        """Merge a branch into a subdirectory of an existing directory."""
        project_wt, lib_wt = self.setup_two_branches()
        self.do_merge_into('lib1', 'project/dir/lib1')
        project_wt.lock_read()
        self.addCleanup(project_wt.unlock)
        self.assertEqual([b'r1-project', b'r1-lib1'], project_wt.get_parent_ids())
        self.assertTreeEntriesEqual([('', b'project-root-id'), ('README', b'project-README-id'), ('dir', b'project-dir-id'), ('dir/file.c', b'project-file.c-id'), ('dir/lib1', b'lib1-root-id'), ('dir/lib1/Makefile', b'lib1-Makefile-id'), ('dir/lib1/README', b'lib1-README-id'), ('dir/lib1/foo.c', b'lib1-foo.c-id')], project_wt)

    def test_newdir_with_repeat_roots(self):
        """If the file-id of the dir to be merged already exists a new ID will
        be allocated to let the merge happen.
        """
        project_wt, lib_wt = self.setup_two_branches(custom_root_ids=False)
        root_id = project_wt.path2id('')
        self.do_merge_into('lib1', 'project/lib1')
        project_wt.lock_read()
        self.addCleanup(project_wt.unlock)
        self.assertEqual([b'r1-project', b'r1-lib1'], project_wt.get_parent_ids())
        new_lib1_id = project_wt.path2id('lib1')
        self.assertNotEqual(None, new_lib1_id)
        self.assertTreeEntriesEqual([('', root_id), ('README', b'project-README-id'), ('dir', b'project-dir-id'), ('lib1', new_lib1_id), ('dir/file.c', b'project-file.c-id'), ('lib1/Makefile', b'lib1-Makefile-id'), ('lib1/README', b'lib1-README-id'), ('lib1/foo.c', b'lib1-foo.c-id')], project_wt)

    def test_name_conflict(self):
        """When the target directory name already exists a conflict is
        generated and the original directory is renamed to foo.moved.
        """
        dest_wt = self.setup_simple_branch('dest', ['dir/', 'dir/file.txt'])
        self.setup_simple_branch('src', ['README'])
        conflicts = self.do_merge_into('src', 'dest/dir')
        self.assertEqual(1, len(conflicts))
        dest_wt.lock_read()
        self.addCleanup(dest_wt.unlock)
        self.assertEqual([b'r1-dest', b'r1-src'], dest_wt.get_parent_ids())
        self.assertTreeEntriesEqual([('', b'dest-root-id'), ('dir', b'src-root-id'), ('dir.moved', b'dest-dir-id'), ('dir/README', b'src-README-id'), ('dir.moved/file.txt', b'dest-file.txt-id')], dest_wt)

    def test_file_id_conflict(self):
        """A conflict is generated if the merge-into adds a file (or other
        inventory entry) with a file-id that already exists in the target tree.
        """
        self.setup_simple_branch('dest', ['file.txt'])
        src_wt = self.make_branch_and_tree('src')
        self.build_tree(['src/README'])
        src_wt.add(['README'], ids=[b'dest-file.txt-id'])
        src_wt.commit('Rev 1 of src.', rev_id=b'r1-src')
        conflicts = self.do_merge_into('src', 'dest/dir')
        self.assertEqual(1, len(conflicts))

    def test_only_subdir(self):
        """When the location points to just part of a tree, merge just that
        subtree.
        """
        dest_wt = self.setup_simple_branch('dest')
        self.setup_simple_branch('src', ['hello.txt', 'dir/', 'dir/foo.c'])
        self.do_merge_into('src/dir', 'dest/dir')
        dest_wt.lock_read()
        self.addCleanup(dest_wt.unlock)
        self.assertEqual([b'r1-dest'], dest_wt.get_parent_ids())
        self.assertTreeEntriesEqual([('', b'dest-root-id'), ('dir', b'src-dir-id'), ('dir/foo.c', b'src-foo.c-id')], dest_wt)

    def test_only_file(self):
        """An edge case: merge just one file, not a whole dir."""
        dest_wt = self.setup_simple_branch('dest')
        self.setup_simple_branch('two-file', ['file1.txt', 'file2.txt'])
        self.do_merge_into('two-file/file1.txt', 'dest/file1.txt')
        dest_wt.lock_read()
        self.addCleanup(dest_wt.unlock)
        self.assertEqual([b'r1-dest'], dest_wt.get_parent_ids())
        self.assertTreeEntriesEqual([('', b'dest-root-id'), ('file1.txt', b'two-file-file1.txt-id')], dest_wt)

    def test_no_such_source_path(self):
        """PathNotInTree is raised if the specified path in the source tree
        does not exist.
        """
        dest_wt = self.setup_simple_branch('dest')
        self.setup_simple_branch('src', ['dir/'])
        self.assertRaises(_mod_merge.PathNotInTree, self.do_merge_into, 'src/no-such-dir', 'dest/foo')
        dest_wt.lock_read()
        self.addCleanup(dest_wt.unlock)
        self.assertEqual([b'r1-dest'], dest_wt.get_parent_ids())
        self.assertTreeEntriesEqual([('', b'dest-root-id')], dest_wt)

    def test_no_such_target_path(self):
        """PathNotInTree is also raised if the specified path in the target
        tree does not exist.
        """
        dest_wt = self.setup_simple_branch('dest')
        self.setup_simple_branch('src', ['file.txt'])
        self.assertRaises(_mod_merge.PathNotInTree, self.do_merge_into, 'src', 'dest/no-such-dir/foo')
        dest_wt.lock_read()
        self.addCleanup(dest_wt.unlock)
        self.assertEqual([b'r1-dest'], dest_wt.get_parent_ids())
        self.assertTreeEntriesEqual([('', b'dest-root-id')], dest_wt)