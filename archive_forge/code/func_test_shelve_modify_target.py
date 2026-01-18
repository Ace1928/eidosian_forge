import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_modify_target(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    tree = self.create_shelvable_tree()
    os.symlink('bar', 'tree/baz')
    tree.add('baz', ids=b'baz-id')
    tree.commit('Add symlink')
    os.unlink('tree/baz')
    os.symlink('vax', 'tree/baz')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
    self.addCleanup(shelver.finalize)
    shelver.expect('Change target of "baz" from "vax" to "bar"?', 0)
    shelver.expect('Apply 1 change(s)?', 0)
    shelver.run()
    self.assertEqual('bar', os.readlink('tree/baz'))