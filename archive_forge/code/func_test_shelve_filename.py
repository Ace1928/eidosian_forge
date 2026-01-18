import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_filename(self):
    tree = self.create_shelvable_tree()
    self.build_tree(['tree/bar'])
    tree.add('bar')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    shelver = ExpectShelver(tree, tree.basis_tree(), file_list=['bar'])
    self.addCleanup(shelver.finalize)
    shelver.expect('Shelve adding file "bar"?', 0)
    shelver.expect('Shelve 1 change(s)?', 0)
    shelver.run()