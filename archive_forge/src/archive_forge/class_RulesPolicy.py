from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.policy import test_backends as policy_tests
class RulesPolicy(unit.TestCase, policy_tests.PolicyTests):

    def setUp(self):
        super(RulesPolicy, self).setUp()
        self.load_backends()

    def config_overrides(self):
        super(RulesPolicy, self).config_overrides()
        self.config_fixture.config(group='policy', driver='rules')

    def test_create(self):
        self.assertRaises(exception.NotImplemented, super(RulesPolicy, self).test_create)

    def test_get(self):
        self.assertRaises(exception.NotImplemented, super(RulesPolicy, self).test_get)

    def test_list(self):
        self.assertRaises(exception.NotImplemented, super(RulesPolicy, self).test_list)

    def test_update(self):
        self.assertRaises(exception.NotImplemented, super(RulesPolicy, self).test_update)

    def test_delete(self):
        self.assertRaises(exception.NotImplemented, super(RulesPolicy, self).test_delete)

    def test_get_policy_returns_not_found(self):
        self.assertRaises(exception.NotImplemented, super(RulesPolicy, self).test_get_policy_returns_not_found)

    def test_update_policy_returns_not_found(self):
        self.assertRaises(exception.NotImplemented, super(RulesPolicy, self).test_update_policy_returns_not_found)

    def test_delete_policy_returns_not_found(self):
        self.assertRaises(exception.NotImplemented, super(RulesPolicy, self).test_delete_policy_returns_not_found)