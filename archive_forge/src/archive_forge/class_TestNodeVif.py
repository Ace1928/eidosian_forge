from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
@mock.patch.object(node.Node, '_get_session', lambda self, x: x)
class TestNodeVif(base.TestCase):

    def setUp(self):
        super(TestNodeVif, self).setUp()
        self.session = mock.Mock(spec=adapter.Adapter)
        self.session.default_microversion = '1.28'
        self.session.log = mock.Mock()
        self.node = node.Node(id='c29db401-b6a7-4530-af8e-20a720dee946', driver=FAKE['driver'])
        self.vif_id = '714bdf6d-2386-4b5e-bd0d-bc036f04b1ef'

    def test_attach_vif(self):
        self.assertIsNone(self.node.attach_vif(self.session, self.vif_id))
        self.session.post.assert_called_once_with('nodes/%s/vifs' % self.node.id, json={'id': self.vif_id}, headers=mock.ANY, microversion='1.28', retriable_status_codes=[409, 503])

    def test_attach_vif_no_retries(self):
        self.assertIsNone(self.node.attach_vif(self.session, self.vif_id, retry_on_conflict=False))
        self.session.post.assert_called_once_with('nodes/%s/vifs' % self.node.id, json={'id': self.vif_id}, headers=mock.ANY, microversion='1.28', retriable_status_codes=[503])

    def test_detach_vif_existing(self):
        self.assertTrue(self.node.detach_vif(self.session, self.vif_id))
        self.session.delete.assert_called_once_with('nodes/%s/vifs/%s' % (self.node.id, self.vif_id), headers=mock.ANY, microversion='1.28', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_detach_vif_missing(self):
        self.session.delete.return_value.status_code = 400
        self.assertFalse(self.node.detach_vif(self.session, self.vif_id))
        self.session.delete.assert_called_once_with('nodes/%s/vifs/%s' % (self.node.id, self.vif_id), headers=mock.ANY, microversion='1.28', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_list_vifs(self):
        self.session.get.return_value.json.return_value = {'vifs': [{'id': '1234'}, {'id': '5678'}]}
        res = self.node.list_vifs(self.session)
        self.assertEqual(['1234', '5678'], res)
        self.session.get.assert_called_once_with('nodes/%s/vifs' % self.node.id, headers=mock.ANY, microversion='1.28')

    def test_incompatible_microversion(self):
        self.session.default_microversion = '1.1'
        self.assertRaises(exceptions.NotSupported, self.node.attach_vif, self.session, self.vif_id)
        self.assertRaises(exceptions.NotSupported, self.node.detach_vif, self.session, self.vif_id)
        self.assertRaises(exceptions.NotSupported, self.node.list_vifs, self.session)