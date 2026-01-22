from openstack.network.v2 import sfc_flow_classifier
from openstack.tests.unit import base
class TestFlowClassifier(base.TestCase):

    def test_basic(self):
        sot = sfc_flow_classifier.SfcFlowClassifier()
        self.assertEqual('flow_classifier', sot.resource_key)
        self.assertEqual('flow_classifiers', sot.resources_key)
        self.assertEqual('/sfc/flow_classifiers', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = sfc_flow_classifier.SfcFlowClassifier(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['protocol'], sot.protocol)
        self.assertEqual(EXAMPLE['ethertype'], sot.ethertype)
        self.assertEqual(EXAMPLE['source_port_range_min'], sot.source_port_range_min)
        self.assertEqual(EXAMPLE['source_port_range_max'], sot.source_port_range_max)
        self.assertEqual(EXAMPLE['destination_port_range_min'], sot.destination_port_range_min)
        self.assertEqual(EXAMPLE['destination_port_range_max'], sot.destination_port_range_max)
        self.assertEqual(EXAMPLE['source_ip_prefix'], sot.source_ip_prefix)
        self.assertEqual(EXAMPLE['destination_ip_prefix'], sot.destination_ip_prefix)
        self.assertEqual(EXAMPLE['logical_source_port'], sot.logical_source_port)
        self.assertEqual(EXAMPLE['logical_destination_port'], sot.logical_destination_port)
        self.assertEqual(EXAMPLE['l7_parameters'], sot.l7_parameters)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'description': 'description', 'name': 'name', 'project_id': 'project_id', 'tenant_id': 'tenant_id', 'ethertype': 'ethertype', 'protocol': 'protocol', 'source_port_range_min': 'source_port_range_min', 'source_port_range_max': 'source_port_range_max', 'destination_port_range_min': 'destination_port_range_min', 'destination_port_range_max': 'destination_port_range_max', 'logical_source_port': 'logical_source_port', 'logical_destination_port': 'logical_destination_port'}, sot._query_mapping._mapping)