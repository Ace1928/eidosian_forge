from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_undotted_member(self):
    self.assertEqual(('mod_name', None, 'attr1'), calc_parent_name('mod_name', 'attr1'))