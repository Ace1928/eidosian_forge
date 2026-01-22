import time
import uuid
from designateclient.tests import v2
class TestZoneImports(v2.APIV2TestCase, v2.CrudMixin):

    def new_ref(self, **kwargs):
        ref = super().new_ref(**kwargs)
        ref.setdefault('zone_id', uuid.uuid4().hex)
        ref.setdefault('created_at', time.strftime('%c'))
        ref.setdefault('updated_at', time.strftime('%c'))
        ref.setdefault('status', 'PENDING')
        ref.setdefault('message', 'Importing...')
        ref.setdefault('version', '1')
        return ref

    def test_create_import(self):
        zonefile = '$ORIGIN example.com'
        parts = ['zones', 'tasks', 'imports']
        self.stub_url('POST', parts=parts, json=zonefile)
        self.client.zone_imports.create(zonefile)
        self.assertRequestBodyIs(body=zonefile)

    def test_get_import(self):
        ref = self.new_ref()
        parts = ['zones', 'tasks', 'imports', ref['id']]
        self.stub_url('GET', parts=parts, json=ref)
        self.stub_entity('GET', parts=parts, entity=ref, id=ref['id'])
        response = self.client.zone_imports.get_import_record(ref['id'])
        self.assertEqual(ref, response)

    def test_list_imports(self):
        items = [self.new_ref(), self.new_ref()]
        parts = ['zones', 'tasks', 'imports']
        self.stub_url('GET', parts=parts, json={'imports': items})
        listed = self.client.zone_imports.list()
        self.assertList(items, listed['imports'])
        self.assertQueryStringIs('')

    def test_delete_import(self):
        ref = self.new_ref()
        parts = ['zones', 'tasks', 'imports', ref['id']]
        self.stub_url('DELETE', parts=parts, json=ref)
        self.stub_entity('DELETE', parts=parts, id=ref['id'])
        self.client.zone_imports.delete(ref['id'])
        self.assertRequestBodyIs(None)