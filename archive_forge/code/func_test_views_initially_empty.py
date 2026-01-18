from breezy import views as _mod_views
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_views_initially_empty(self):
    wt = self.make_branch_and_tree('wt')
    current, views = wt.views.get_view_info()
    self.assertEqual(None, current)
    self.assertEqual({}, views)