import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_all(self):
    self.create_shelvable_tree()
    shelver = ExpectShelver.from_args(sys.stdout, all=True, directory='tree')
    try:
        shelver.run()
    finally:
        shelver.finalize()
    self.assertFileEqual(LINES_AJ, 'tree/foo')