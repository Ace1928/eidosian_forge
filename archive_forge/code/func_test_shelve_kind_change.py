import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_kind_change(self):
    tree = self.create_shelvable_tree()
    os.unlink('tree/foo')
    os.mkdir('tree/foo')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
    self.addCleanup(shelver.finalize)
    shelver.expect('Change "foo" from directory to a file?', 0)
    shelver.expect('Apply 1 change(s)?', 0)