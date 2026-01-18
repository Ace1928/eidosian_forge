import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_unshelve_args_preview(self):
    tree = self.create_tree_with_shelf()
    write_diff_to = BytesIO()
    unshelver = shelf_ui.Unshelver.from_args(directory='tree', action='preview', write_diff_to=write_diff_to)
    try:
        unshelver.run()
    finally:
        unshelver.tree.unlock()
    self.assertFileEqual(LINES_AJ, 'tree/foo')
    self.assertEqual(1, tree.get_shelf_manager().last_shelf())
    diff = write_diff_to.getvalue()
    expected = dedent('            @@ -1,4 +1,4 @@\n            -a\n            +z\n             b\n             c\n             d\n            @@ -7,4 +7,4 @@\n             g\n             h\n             i\n            -j\n            +y\n\n            ')
    self.assertEqualDiff(expected.encode('utf-8'), diff[-len(expected):])