import shutil
from breezy import errors
from breezy.tests import TestNotApplicable, TestSkipped, features, per_tree
from breezy.tree import MissingNestedTree
def _create_tree_with_utf8(self, tree):
    self.requireFeature(features.UnicodeFilenameFeature)
    paths = ['', 'fo€o', 'ba€r/', 'ba€r/ba€z']
    file_ids = [b'TREE_ROOT', b'fo\xe2\x82\xaco-id', b'ba\xe2\x82\xacr-id', b'ba\xe2\x82\xacz-id']
    self.build_tree(paths[1:])
    if not tree.is_versioned(''):
        tree.add(paths, ids=file_ids)
    elif tree.supports_setting_file_ids():
        tree.set_root_id(file_ids[0])
        tree.add(paths[1:], ids=file_ids[1:])
    else:
        tree.add(paths[1:])
    if tree.branch.repository._format.supports_setting_revision_ids:
        try:
            tree.commit('inítial', rev_id='rév-1'.encode())
        except errors.NonAsciiRevisionId:
            raise TestSkipped('non-ascii revision ids not supported')
    else:
        tree.commit('inítial')
    return tree