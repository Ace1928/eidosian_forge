import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
class TestReportChanges(tests.TestCase):
    """Test the new change reporting infrastructure"""

    def assertReport(self, expected, file_id=b'fid', path='path', versioned_change='unchanged', renamed=False, copied=False, modified='unchanged', exe_change=False, kind=('file', 'file'), old_path=None, unversioned_filter=None, view_info=None):
        if expected is None:
            expected_lines = None
        else:
            expected_lines = [expected]
        self.assertReportLines(expected_lines, file_id, path, versioned_change, renamed, copied, modified, exe_change, kind, old_path, unversioned_filter, view_info)

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

    def test_rename(self):
        self.assertReport('R   old => path', renamed=True, old_path='old')
        self.assertReport('    path')
        self.assertReport('RN  old => path', renamed=True, old_path='old', modified='created', kind=(None, 'file'))

    def test_kind(self):
        self.assertReport(' K  path => path/', modified='kind changed', kind=('file', 'directory'), old_path='path')
        self.assertReport(' K  path/ => path', modified='kind changed', kind=('directory', 'file'), old_path='old')
        self.assertReport('RK  old => path/', renamed=True, modified='kind changed', kind=('file', 'directory'), old_path='old')

    def test_new(self):
        self.assertReport(' N  path/', modified='created', kind=(None, 'directory'))
        self.assertReport('+   path/', versioned_change='added', modified='unchanged', kind=(None, 'directory'))
        self.assertReport('+   path', versioned_change='added', modified='unchanged', kind=(None, None))
        self.assertReport('+N  path/', versioned_change='added', modified='created', kind=(None, 'directory'))
        self.assertReport('+M  path/', versioned_change='added', modified='modified', kind=(None, 'directory'))

    def test_removal(self):
        self.assertReport(' D  path/', modified='deleted', kind=('directory', None), old_path='old')
        self.assertReport('-   path/', versioned_change='removed', old_path='path', kind=(None, 'directory'))
        self.assertReport('-D  path', versioned_change='removed', old_path='path', modified='deleted', kind=('file', 'directory'))

    def test_modification(self):
        self.assertReport(' M  path', modified='modified')
        self.assertReport(' M* path', modified='modified', exe_change=True)

    def test_unversioned(self):
        self.assertReport('?   subdir/foo~', file_id=None, path='subdir/foo~', old_path=None, versioned_change='unversioned', renamed=False, modified='created', exe_change=False, kind=(None, 'file'))
        self.assertReport(None, file_id=None, path='subdir/foo~', old_path=None, versioned_change='unversioned', renamed=False, modified='created', exe_change=False, kind=(None, 'file'), unversioned_filter=lambda x: True)

    def test_missing(self):
        self.assertReport('+!  missing.c', file_id=None, path='missing.c', old_path=None, versioned_change='added', renamed=False, modified='missing', exe_change=False, kind=(None, None))

    def test_view_filtering(self):
        expected_lines = ["Operating on whole tree but only reporting on 'my' view.", ' M  path']
        self.assertReportLines(expected_lines, modified='modified', view_info=('my', ['path']))
        expected_lines = ["Operating on whole tree but only reporting on 'my' view."]
        self.assertReportLines(expected_lines, modified='modified', path='foo', view_info=('my', ['path']))

    def assertChangesEqual(self, file_id=b'fid', paths=('path', 'path'), content_change=False, versioned=(True, True), parent_id=('pid', 'pid'), name=('name', 'name'), kind=('file', 'file'), executable=(False, False), versioned_change='unchanged', renamed=False, copied=False, modified='unchanged', exe_change=False):
        reporter = InstrumentedReporter()
        _mod_delta.report_changes([InventoryTreeChange(file_id, paths, content_change, versioned, parent_id, name, kind, executable, copied)], reporter)
        output = reporter.calls[0]
        self.assertEqual(paths, output[0])
        self.assertEqual(versioned_change, output[1])
        self.assertEqual(renamed, output[2])
        self.assertEqual(copied, output[3])
        self.assertEqual(modified, output[4])
        self.assertEqual(exe_change, output[5])
        self.assertEqual(kind, output[6])

    def test_report_changes(self):
        """Test change detection of report_changes"""
        self.assertChangesEqual(modified='unchanged', renamed=False, versioned_change='unchanged', exe_change=False)
        self.assertChangesEqual(modified='kind changed', kind=('file', 'directory'))
        self.assertChangesEqual(modified='created', kind=(None, 'directory'))
        self.assertChangesEqual(modified='deleted', kind=('directory', None))
        self.assertChangesEqual(content_change=True, modified='modified')
        self.assertChangesEqual(renamed=True, name=('old', 'new'))
        self.assertChangesEqual(renamed=True, parent_id=('old-parent', 'new-parent'))
        self.assertChangesEqual(versioned_change='added', versioned=(False, True))
        self.assertChangesEqual(versioned_change='removed', versioned=(True, False))
        self.assertChangesEqual(exe_change=True, executable=(True, False))
        self.assertChangesEqual(exe_change=False, executable=(True, False), kind=('directory', 'directory'))
        self.assertChangesEqual(exe_change=False, modified='kind changed', executable=(False, True), kind=('directory', 'file'))
        self.assertChangesEqual(parent_id=('pid', None))
        self.assertChangesEqual(versioned_change='removed', modified='deleted', versioned=(True, False), kind=('directory', None))
        self.assertChangesEqual(versioned_change='removed', modified='created', versioned=(True, False), kind=(None, 'file'))
        self.assertChangesEqual(versioned_change='removed', modified='modified', renamed=True, exe_change=True, versioned=(True, False), content_change=True, name=('old', 'new'), executable=(False, True))

    def test_report_unversioned(self):
        """Unversioned entries are reported well."""
        self.assertChangesEqual(file_id=None, paths=(None, 'full/path'), content_change=True, versioned=(False, False), parent_id=(None, None), name=(None, 'path'), kind=(None, 'file'), executable=(None, False), versioned_change='unversioned', renamed=False, modified='created', exe_change=False)