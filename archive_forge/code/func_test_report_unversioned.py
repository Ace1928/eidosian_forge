import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
def test_report_unversioned(self):
    """Unversioned entries are reported well."""
    self.assertChangesEqual(file_id=None, paths=(None, 'full/path'), content_change=True, versioned=(False, False), parent_id=(None, None), name=(None, 'path'), kind=(None, 'file'), executable=(None, False), versioned_change='unversioned', renamed=False, modified='created', exe_change=False)