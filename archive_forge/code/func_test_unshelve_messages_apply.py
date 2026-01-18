import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_unshelve_messages_apply(self):
    self.create_tree_with_shelf()
    self.run_script('\n$ cd tree\n$ brz unshelve --apply\n2>Using changes with id "1".\n2> M  foo\n2>All changes applied successfully.\n2>Deleted changes with id "1".\n')