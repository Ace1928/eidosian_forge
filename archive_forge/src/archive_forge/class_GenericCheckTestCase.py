from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
class GenericCheckTestCase(base.PolicyBaseTestCase):

    def test_no_cred(self):
        check = _checks.GenericCheck('name', '%(name)s')
        self.assertFalse(check(dict(name='spam'), {}, self.enforcer))

    def test_cred_mismatch(self):
        check = _checks.GenericCheck('name', '%(name)s')
        self.assertFalse(check(dict(name='spam'), dict(name='ham'), self.enforcer))

    def test_accept(self):
        check = _checks.GenericCheck('name', '%(name)s')
        self.assertTrue(check(dict(name='spam'), dict(name='spam'), self.enforcer))

    def test_no_key_match_in_target(self):
        check = _checks.GenericCheck('name', '%(name)s')
        self.assertFalse(check(dict(name1='spam'), dict(name='spam'), self.enforcer))

    def test_constant_string_mismatch(self):
        check = _checks.GenericCheck("'spam'", '%(name)s')
        self.assertFalse(check(dict(name='ham'), {}, self.enforcer))

    def test_constant_string_accept(self):
        check = _checks.GenericCheck("'spam'", '%(name)s')
        self.assertTrue(check(dict(name='spam'), {}, self.enforcer))

    def test_constant_literal_mismatch(self):
        check = _checks.GenericCheck('True', '%(enabled)s')
        self.assertFalse(check(dict(enabled=False), {}, self.enforcer))

    def test_constant_literal_accept(self):
        check = _checks.GenericCheck('True', '%(enabled)s')
        self.assertTrue(check(dict(enabled=True), {}, self.enforcer))

    def test_deep_credentials_dictionary_lookup(self):
        check = _checks.GenericCheck('a.b.c.d', 'APPLES')
        credentials = {'a': {'b': {'c': {'d': 'APPLES'}}}}
        self.assertTrue(check({}, credentials, self.enforcer))

    def test_missing_credentials_dictionary_lookup(self):
        credentials = {'a': 'APPLES', 'o': {'t': 'ORANGES'}}
        check = _checks.GenericCheck('o.t', 'ORANGES')
        self.assertTrue(check({}, credentials, self.enforcer))
        check = _checks.GenericCheck('o.v', 'ORANGES')
        self.assertFalse(check({}, credentials, self.enforcer))
        check = _checks.GenericCheck('q.v', 'APPLES')
        self.assertFalse(check({}, credentials, self.enforcer))

    def test_single_entry_in_list_accepted(self):
        check = _checks.GenericCheck('a.b.c.d', 'APPLES')
        credentials = {'a': {'b': {'c': {'d': ['APPLES']}}}}
        self.assertTrue(check({}, credentials, self.enforcer))

    def test_multiple_entry_in_list_accepted(self):
        check = _checks.GenericCheck('a.b.c.d', 'APPLES')
        credentials = {'a': {'b': {'c': {'d': ['Bananas', 'APPLES', 'Grapes']}}}}
        self.assertTrue(check({}, credentials, self.enforcer))

    def test_multiple_entry_in_nested_list_accepted(self):
        check = _checks.GenericCheck('a.b.c.d', 'APPLES')
        credentials = {'a': {'b': [{'c': {'d': ['BANANAS', 'APPLES', 'GRAPES']}}, {}]}}
        self.assertTrue(check({}, credentials, self.enforcer))

    def test_multiple_entries_one_matches(self):
        check = _checks.GenericCheck('token.catalog.endpoints.id', token_fixture.REGION_ONE_PUBLIC_KEYSTONE_ENDPOINT_ID)
        credentials = token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE
        self.assertTrue(check({}, credentials, self.enforcer))

    def test_generic_role_check_matches(self):
        check = _checks.GenericCheck('token.roles.name', 'role1')
        credentials = token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE
        self.assertTrue(check({}, credentials, self.enforcer))

    def test_generic_missing_role_does_not_matches(self):
        check = _checks.GenericCheck('token.roles.name', 'missing')
        credentials = token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE
        self.assertFalse(check({}, credentials, self.enforcer))

    def test_multiple_nested_lists_accepted(self):
        check = _checks.GenericCheck('a.b.c.d', 'APPLES')
        credentials = {'a': {'b': [{'a': ''}, {'c': {'d': ['BANANAS', 'APPLES', 'GRAPES']}}, {}]}}
        self.assertTrue(check({}, credentials, self.enforcer))

    def test_entry_not_in_list_rejected(self):
        check = _checks.GenericCheck('a.b.c.d', 'APPLES')
        credentials = {'a': {'b': {'c': {'d': ['PEACHES', 'PEARS']}}}}
        self.assertFalse(check({}, credentials, self.enforcer))