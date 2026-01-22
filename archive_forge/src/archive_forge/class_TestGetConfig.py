from breezy import branch, errors, tests
from breezy.tests import per_branch
class TestGetConfig(per_branch.TestCaseWithBranch):

    def test_set_user_option_with_dict(self):
        b = self.make_branch('b')
        config = b.get_config()
        value_dict = {'ascii': 'abcd', 'unicode ⌚': 'foo ‽'}
        config.set_user_option('name', value_dict.copy())
        self.assertEqual(value_dict, config.get_user_option('name'))

    def test_set_submit_branch(self):
        b = self.make_branch('.')
        b.set_submit_branch('foo')
        b = branch.Branch.open('.')
        self.assertEqual('foo', b.get_submit_branch())