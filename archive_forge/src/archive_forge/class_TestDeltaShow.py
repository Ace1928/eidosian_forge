import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
class TestDeltaShow(tests.TestCaseWithTransport):

    def _get_delta(self):
        wt = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/f1', b'1\n'), ('branch/f2', b'2\n'), ('branch/f3', b'3\n'), ('branch/f4', b'4\n'), ('branch/f5', b'5\n'), ('branch/dir/',)])
        wt.add(['f1', 'f2', 'f3', 'f4', 'dir'], ids=[b'f1-id', b'f2-id', b'f3-id', b'f4-id', b'dir-id'])
        wt.commit('commit one', rev_id=b'1')
        wt.add('f5')
        os.unlink('branch/f5')
        long_status = 'added:\n  dir/\n  f1\n  f2\n  f3\n  f4\nmissing:\n  f5\n'
        short_status = 'A  dir/\nA  f1\nA  f2\nA  f3\nA  f4\n!  f5\n'
        repo = wt.branch.repository
        d = wt.changes_from(repo.revision_tree(_mod_revision.NULL_REVISION))
        return (d, long_status, short_status)

    def test_short_status(self):
        d, long_status, short_status = self._get_delta()
        out = StringIO()
        _mod_delta.report_delta(out, d, short_status=True)
        self.assertEqual(short_status, out.getvalue())

    def test_long_status(self):
        d, long_status, short_status = self._get_delta()
        out = StringIO()
        _mod_delta.report_delta(out, d, short_status=False)
        self.assertEqual(long_status, out.getvalue())

    def test_predicate_always(self):
        d, long_status, short_status = self._get_delta()
        out = StringIO()

        def always(path):
            return True
        _mod_delta.report_delta(out, d, short_status=True, predicate=always)
        self.assertEqual(short_status, out.getvalue())

    def test_short_status_path_predicate(self):
        d, long_status, short_status = self._get_delta()
        out = StringIO()

        def only_f2(path):
            return path == 'f2'
        _mod_delta.report_delta(out, d, short_status=True, predicate=only_f2)
        self.assertEqual('A  f2\n', out.getvalue())

    def test_long_status_path_predicate(self):
        d, long_status, short_status = self._get_delta()
        out = StringIO()

        def only_f2(path):
            return path == 'f2'
        _mod_delta.report_delta(out, d, short_status=False, predicate=only_f2)
        self.assertEqual('added:\n  f2\n', out.getvalue())