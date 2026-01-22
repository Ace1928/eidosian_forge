from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import trunk
from openstack.tests.unit import base
class TestTrunk(base.TestCase):

    def test_basic(self):
        sot = trunk.Trunk()
        self.assertEqual('trunk', sot.resource_key)
        self.assertEqual('trunks', sot.resources_key)
        self.assertEqual('/trunks', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = trunk.Trunk(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['admin_state_up'], sot.is_admin_state_up)
        self.assertEqual(EXAMPLE['port_id'], sot.port_id)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['sub_ports'], sot.sub_ports)

    def test_add_subports_4xx(self):
        sot = trunk.Trunk(**EXAMPLE)
        response = mock.Mock()
        msg = '.*borked'
        response.body = {'NeutronError': {'message': msg}}
        response.json = mock.Mock(return_value=response.body)
        response.ok = False
        response.status_code = 404
        response.headers = {'content-type': 'application/json'}
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        subports = [{'port_id': 'abc', 'segmentation_id': '123', 'segmentation_type': 'vlan'}]
        with testtools.ExpectedException(exceptions.ResourceNotFound, msg):
            sot.add_subports(sess, subports)

    def test_delete_subports_4xx(self):
        sot = trunk.Trunk(**EXAMPLE)
        response = mock.Mock()
        msg = '.*borked'
        response.body = {'NeutronError': {'message': msg}}
        response.json = mock.Mock(return_value=response.body)
        response.ok = False
        response.status_code = 404
        response.headers = {'content-type': 'application/json'}
        sess = mock.Mock()
        sess.put = mock.Mock(return_value=response)
        subports = [{'port_id': 'abc', 'segmentation_id': '123', 'segmentation_type': 'vlan'}]
        with testtools.ExpectedException(exceptions.ResourceNotFound, msg):
            sot.delete_subports(sess, subports)