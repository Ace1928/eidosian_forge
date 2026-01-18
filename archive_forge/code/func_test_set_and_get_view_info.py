from breezy import views as _mod_views
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_set_and_get_view_info(self):
    wt = self.make_branch_and_tree('wt')
    view_current = 'view-name'
    view_dict = {view_current: ['dir-1'], 'other-name': ['dir-2']}
    wt.views.set_view_info(view_current, view_dict)
    current, views = wt.views.get_view_info()
    self.assertEqual(view_current, current)
    self.assertEqual(view_dict, views)
    wt = WorkingTree.open('wt')
    current, views = wt.views.get_view_info()
    self.assertEqual(view_current, current)
    self.assertEqual(view_dict, views)
    self.assertRaises(_mod_views.NoSuchView, wt.views.set_view_info, 'yet-another', view_dict)
    current, views = wt.views.get_view_info()
    self.assertEqual(view_current, current)
    self.assertEqual(view_dict, views)
    wt.views.set_view_info(None, view_dict)
    current, views = wt.views.get_view_info()
    self.assertEqual(None, current)
    self.assertEqual(view_dict, views)