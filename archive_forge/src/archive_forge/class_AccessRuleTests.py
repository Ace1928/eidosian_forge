import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import access_rules
class AccessRuleTests(utils.ClientTestCase, utils.CrudTests):

    def setUp(self):
        super(AccessRuleTests, self).setUp()
        self.key = 'access_rule'
        self.collection_key = 'access_rules'
        self.model = access_rules.AccessRule
        self.manager = self.client.access_rules
        self.path_prefix = 'users/%s' % self.TEST_USER_ID

    def new_ref(self, **kwargs):
        kwargs = super(AccessRuleTests, self).new_ref(**kwargs)
        kwargs.setdefault('path', uuid.uuid4().hex)
        kwargs.setdefault('method', uuid.uuid4().hex)
        kwargs.setdefault('service', uuid.uuid4().hex)
        return kwargs

    def test_update(self):
        self.assertRaises(exceptions.MethodNotImplemented, self.manager.update)

    def test_create(self):
        self.assertRaises(exceptions.MethodNotImplemented, self.manager.create)