from unittest import mock
from heat.engine.clients.os import cinder as c_plugin
from heat.engine.resources.openstack.cinder import qos_specs
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class QoSAssociationTest(common.HeatTestCase):

    def setUp(self):
        super(QoSAssociationTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.qos_specs_id = 'foobar'
        self.patchobject(c_plugin.CinderClientPlugin, 'has_extension', return_value=True)
        self.patchobject(c_plugin.CinderClientPlugin, 'get_qos_specs', return_value=self.qos_specs_id)
        self.stack = stack.Stack(self.ctx, 'cinder_qos_associate_test_stack', template.Template(QOS_ASSOCIATE_TEMPLATE))
        self.my_qos_associate = self.stack['my_qos_associate']
        cinder_client = mock.MagicMock()
        self.cinderclient = mock.MagicMock()
        self.my_qos_associate.client = cinder_client
        cinder_client.return_value = self.cinderclient
        self.qos_specs = self.cinderclient.qos_specs
        self.stub_QoSSpecsConstraint_validate()
        self.stub_VolumeTypeConstraint_validate()
        self.vt_ceph = 'ceph'
        self.vt_lvm = 'lvm'
        self.vt_new_ceph = 'new_ceph'

    def test_resource_mapping(self):
        mapping = qos_specs.resource_mapping()
        self.assertEqual(2, len(mapping))
        self.assertEqual(qos_specs.QoSAssociation, mapping['OS::Cinder::QoSAssociation'])
        self.assertIsInstance(self.my_qos_associate, qos_specs.QoSAssociation)

    def _set_up_qos_associate_environment(self):
        self.my_qos_associate.handle_create()

    def test_qos_associate_handle_create(self):
        self.patchobject(c_plugin.CinderClientPlugin, 'get_volume_type', side_effect=[self.vt_ceph, self.vt_lvm])
        self._set_up_qos_associate_environment()
        self.cinderclient.qos_specs.associate.assert_any_call(self.qos_specs_id, self.vt_ceph)
        self.qos_specs.associate.assert_any_call(self.qos_specs_id, self.vt_lvm)

    def test_qos_associate_handle_update(self):
        self.patchobject(c_plugin.CinderClientPlugin, 'get_volume_type', side_effect=[self.vt_lvm, self.vt_ceph, self.vt_new_ceph, self.vt_ceph])
        self._set_up_qos_associate_environment()
        prop_diff = {'volume_types': [self.vt_lvm, self.vt_new_ceph]}
        self.my_qos_associate.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
        self.qos_specs.associate.assert_any_call(self.qos_specs_id, self.vt_new_ceph)
        self.qos_specs.disassociate.assert_any_call(self.qos_specs_id, self.vt_ceph)

    def test_qos_associate_handle_delete_specs(self):
        self.patchobject(c_plugin.CinderClientPlugin, 'get_volume_type', side_effect=[self.vt_ceph, self.vt_lvm, self.vt_ceph, self.vt_lvm])
        self._set_up_qos_associate_environment()
        self.my_qos_associate.handle_delete()
        self.qos_specs.disassociate.assert_any_call(self.qos_specs_id, self.vt_ceph)
        self.qos_specs.disassociate.assert_any_call(self.qos_specs_id, self.vt_lvm)