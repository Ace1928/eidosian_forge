from openstack.dns.v2 import recordset
from openstack.tests.unit import base
class TestRecordset(base.TestCase):

    def test_basic(self):
        sot = recordset.Recordset()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('recordsets', sot.resources_key)
        self.assertEqual('/zones/%(zone_id)s/recordsets', sot.base_path)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertDictEqual({'data': 'data', 'description': 'description', 'limit': 'limit', 'marker': 'marker', 'name': 'name', 'status': 'status', 'ttl': 'ttl', 'type': 'type'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = recordset.Recordset(**EXAMPLE)
        self.assertEqual(IDENTIFIER, sot.id)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['ttl'], sot.ttl)
        self.assertEqual(EXAMPLE['type'], sot.type)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['status'], sot.status)