import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_unshelve_args(self):
    tree = self.create_tree_with_shelf()
    unshelver = shelf_ui.Unshelver.from_args(directory='tree')
    try:
        unshelver.run()
    finally:
        unshelver.tree.unlock()
    self.assertFileEqual(LINES_ZY, 'tree/foo')
    self.assertIs(None, tree.get_shelf_manager().last_shelf())