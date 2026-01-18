import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_unshelve(self):
    tree = self.create_tree_with_shelf()
    tree.lock_write()
    self.addCleanup(tree.unlock)
    manager = tree.get_shelf_manager()
    shelf_ui.Unshelver(tree, manager, 1, True, True, True).run()
    self.assertFileEqual(LINES_ZY, 'tree/foo')