import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_wrong_prompt_failure(self):
    tree = self.create_shelvable_tree()
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    shelver = ExpectShelver(tree, tree.basis_tree())
    self.addCleanup(shelver.finalize)
    shelver.expect('foo', 0)
    e = self.assertRaises(AssertionError, shelver.run)
    self.assertEqual('Wrong prompt: Shelve?', str(e))