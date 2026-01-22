import uuid
from designateclient.tests import v2
class TestBlacklists(v2.APIV2TestCase, v2.CrudMixin):
    RESOURCE = 'blacklists'

    def new_ref(self, **kwargs):
        ref = super().new_ref(**kwargs)
        ref.setdefault('pattern', uuid.uuid4().hex)
        return ref

    def test_create(self):
        ref = self.new_ref()
        self.stub_url('POST', parts=[self.RESOURCE], json=ref)
        values = ref.copy()
        del values['id']
        self.client.blacklists.create(**values)
        self.assertRequestBodyIs(json=values)

    def test_create_with_description(self):
        ref = self.new_ref(description='My Blacklist')
        self.stub_url('POST', parts=[self.RESOURCE], json=ref)
        values = ref.copy()
        del values['id']
        self.client.blacklists.create(**values)
        self.assertRequestBodyIs(json=values)

    def test_get(self):
        ref = self.new_ref()
        self.stub_entity('GET', entity=ref, id=ref['id'])
        response = self.client.blacklists.get(ref['id'])
        self.assertEqual(ref, response)

    def test_list(self):
        items = [self.new_ref(), self.new_ref()]
        self.stub_url('GET', parts=[self.RESOURCE], json={'blacklists': items})
        listed = self.client.blacklists.list()
        self.assertList(items, listed)
        self.assertQueryStringIs('')

    def test_update(self):
        ref = self.new_ref()
        self.stub_entity('PATCH', entity=ref, id=ref['id'])
        values = ref.copy()
        del values['id']
        self.client.blacklists.update(ref['id'], values)
        self.assertRequestBodyIs(json=values)

    def test_delete(self):
        ref = self.new_ref()
        self.stub_entity('DELETE', id=ref['id'])
        self.client.blacklists.delete(ref['id'])
        self.assertRequestBodyIs(None)