import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
def assertReportLines(self, expected_lines, file_id=b'fid', path='path', versioned_change='unchanged', renamed=False, copied=False, modified='unchanged', exe_change=False, kind=('file', 'file'), old_path=None, unversioned_filter=None, view_info=None):
    result = []

    def result_line(format, *args):
        result.append(format % args)
    reporter = _mod_delta._ChangeReporter(result_line, unversioned_filter=unversioned_filter, view_info=view_info)
    reporter.report((old_path, path), versioned_change, renamed, copied, modified, exe_change, kind)
    if expected_lines is not None:
        self.assertEqualDiff('\n'.join(expected_lines), '\n'.join(result))
    else:
        self.assertEqual([], result)