from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_package_attr(self):
    self.assertIs(tests.TestCase, get_named_object('breezy.tests', 'TestCase'))