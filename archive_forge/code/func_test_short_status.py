import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
def test_short_status(self):
    d, long_status, short_status = self._get_delta()
    out = StringIO()
    _mod_delta.report_delta(out, d, short_status=True)
    self.assertEqual(short_status, out.getvalue())