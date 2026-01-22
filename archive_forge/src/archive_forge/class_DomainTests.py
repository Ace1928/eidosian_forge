import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import domains
class DomainTests(utils.ClientTestCase, utils.CrudTests):

    def setUp(self):
        super(DomainTests, self).setUp()
        self.key = 'domain'
        self.collection_key = 'domains'
        self.model = domains.Domain
        self.manager = self.client.domains

    def new_ref(self, **kwargs):
        kwargs = super(DomainTests, self).new_ref(**kwargs)
        kwargs.setdefault('enabled', True)
        kwargs.setdefault('name', uuid.uuid4().hex)
        return kwargs

    def test_filter_for_default_domain_by_id(self):
        ref = self.new_ref(id='default')
        super(DomainTests, self).test_list_by_id(ref=ref, id=ref['id'])

    def test_list_filter_name(self):
        super(DomainTests, self).test_list(name='adomain123')

    def test_list_filter_enabled(self):
        super(DomainTests, self).test_list(enabled=True)

    def test_list_filter_disabled(self):
        expected_query = {'enabled': '0'}
        super(DomainTests, self).test_list(expected_query=expected_query, enabled=False)

    def test_update_enabled_defaults_to_none(self):
        super(DomainTests, self).test_update(req_ref={'name': uuid.uuid4().hex})