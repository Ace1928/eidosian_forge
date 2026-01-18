import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
def test_short_status_path_predicate(self):
    d, long_status, short_status = self._get_delta()
    out = StringIO()

    def only_f2(path):
        return path == 'f2'
    _mod_delta.report_delta(out, d, short_status=True, predicate=only_f2)
    self.assertEqual('A  f2\n', out.getvalue())