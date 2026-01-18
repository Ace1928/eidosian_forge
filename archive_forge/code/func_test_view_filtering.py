import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
def test_view_filtering(self):
    expected_lines = ["Operating on whole tree but only reporting on 'my' view.", ' M  path']
    self.assertReportLines(expected_lines, modified='modified', view_info=('my', ['path']))
    expected_lines = ["Operating on whole tree but only reporting on 'my' view."]
    self.assertReportLines(expected_lines, modified='modified', path='foo', view_info=('my', ['path']))