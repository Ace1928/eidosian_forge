from breezy import views as _mod_views
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_unicode_view(self):
    wt = self.make_branch_and_tree('wt')
    view_name = 'ば'
    view_files = ['foo', 'bar/']
    view_dict = {view_name: view_files}
    wt.views.set_view_info(view_name, view_dict)
    current, views = wt.views.get_view_info()
    self.assertEqual(view_name, current)
    self.assertEqual(view_dict, views)