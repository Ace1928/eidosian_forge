import shutil
from breezy import errors
from breezy.tests import TestNotApplicable, TestSkipped, features, per_tree
from breezy.tree import MissingNestedTree
def test_tree_with_utf8(self):
    tree = self.make_branch_and_tree('.')
    if not tree.supports_setting_file_ids():
        raise TestNotApplicable('format does not support custom file ids')
    self._create_tree_with_utf8(tree)
    tree = self.workingtree_to_test_tree(tree)
    revision_id = 'rév-1'.encode()
    root_id = b'TREE_ROOT'
    bar_id = 'ba€r-id'.encode()
    foo_id = 'fo€o-id'.encode()
    baz_id = 'ba€z-id'.encode()
    path_and_ids = [('', root_id, None, None), ('ba€r', bar_id, root_id, revision_id), ('fo€o', foo_id, root_id, revision_id), ('ba€r/ba€z', baz_id, bar_id, revision_id)]
    with tree.lock_read():
        path_entries = list(tree.iter_entries_by_dir())
    for expected, (path, ie) in zip(path_and_ids, path_entries):
        self.assertEqual(expected[0], path)
        self.assertIsInstance(path, str)
        self.assertEqual(expected[1], ie.file_id)
        self.assertIsInstance(ie.file_id, bytes)
        self.assertEqual(expected[2], ie.parent_id)
        if expected[2] is not None:
            self.assertIsInstance(ie.parent_id, bytes)
        if ie.revision is not None:
            self.assertIsInstance(ie.revision, bytes)
            if expected[0] != '':
                self.assertEqual(revision_id, ie.revision)
    self.assertEqual(len(path_and_ids), len(path_entries))
    get_revision_id = getattr(tree, 'get_revision_id', None)
    if get_revision_id is not None:
        self.assertIsInstance(get_revision_id(), bytes)
    last_revision = getattr(tree, 'last_revision', None)
    if last_revision is not None:
        self.assertIsInstance(last_revision(), bytes)