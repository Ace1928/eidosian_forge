import shutil
from breezy import errors
from breezy.tests import TestNotApplicable, TestSkipped, features, per_tree
from breezy.tree import MissingNestedTree
def test_tree_with_subdirs_and_all_content_types_wo_symlinks(self):
    tree = self.get_tree_with_subdirs_and_all_supported_content_types(False)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual([], tree.get_parent_ids())
    self.assertEqual([], tree.conflicts())
    self.assertEqual([], list(tree.unknowns()))
    if tree.has_versioned_directories():
        self.assertEqual({'', '0file', '1top-dir', '1top-dir/0file-in-1topdir', '1top-dir/1dir-in-1topdir', '2utfሴfile'}, set(tree.all_versioned_paths()))
        self.assertEqual([('', 'directory'), ('0file', 'file'), ('1top-dir', 'directory'), ('2utfሴfile', 'file'), ('1top-dir/0file-in-1topdir', 'file'), ('1top-dir/1dir-in-1topdir', 'directory')], [(path, node.kind) for path, node in tree.iter_entries_by_dir()])
    else:
        self.assertEqual({'', '0file', '1top-dir', '1top-dir/0file-in-1topdir', '2utfሴfile'}, set(tree.all_versioned_paths()))
        self.assertEqual([('', 'directory'), ('0file', 'file'), ('1top-dir', 'directory'), ('2utfሴfile', 'file'), ('1top-dir/0file-in-1topdir', 'file')], [(path, node.kind) for path, node in tree.iter_entries_by_dir()])