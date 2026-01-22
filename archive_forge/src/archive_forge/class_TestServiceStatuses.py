from designateclient.tests import v2
class TestServiceStatuses(v2.APIV2TestCase, v2.CrudMixin):
    RESOURCE = 'service_statuses'

    def new_ref(self, **kwargs):
        ref = super().new_ref(**kwargs)
        ref['name'] = 'foo'
        return ref

    def test_get(self):
        ref = self.new_ref()
        self.stub_entity('GET', entity=ref, id=ref['id'])
        response = self.client.service_statuses.get(ref['id'])
        self.assertEqual(ref, response)

    def test_list(self):
        items = [self.new_ref(), self.new_ref()]
        self.stub_url('GET', parts=[self.RESOURCE], json={'service_statuses': items})
        listed = self.client.service_statuses.list()
        self.assertList(items, listed)
        self.assertQueryStringIs('')