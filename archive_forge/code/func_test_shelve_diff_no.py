import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_diff_no(self):
    tree = self.create_shelvable_tree()
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
    self.addCleanup(shelver.finalize)
    shelver.expect('Apply change?', 0)
    shelver.expect('Apply change?', 0)
    shelver.expect('Apply 2 change(s)?', 1)
    shelver.run()
    self.assertFileEqual(LINES_ZY, 'tree/foo')