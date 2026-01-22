import sys
from io import BytesIO
from ... import rules, status
from ...workingtree import WorkingTree
from .. import TestSkipped
from . import TestCaseWithWorkingTree
def assertNewContentForSetting(self, wt, eol, expected_unix, expected_win, roundtrip):
    """Clone a working tree and check the convenience content.

        If roundtrip is True, status and commit should see no changes.
        """
    if expected_win is None:
        expected_win = expected_unix
    self.patch_rules_searcher(eol)
    wt2 = wt.controldir.sprout('tree-%s' % eol).open_workingtree()
    with wt2.get_file('file1', filtered=False) as f:
        content = f.read()
    if sys.platform == 'win32':
        self.assertEqual(expected_win, content)
    else:
        self.assertEqual(expected_unix, content)
    if roundtrip:
        status_io = BytesIO()
        status.show_tree_status(wt2, to_file=status_io)
        self.assertEqual(b'', status_io.getvalue())