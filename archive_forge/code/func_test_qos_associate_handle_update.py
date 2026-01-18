from unittest import mock
from heat.engine.clients.os import cinder as c_plugin
from heat.engine.resources.openstack.cinder import qos_specs
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_qos_associate_handle_update(self):
    self.patchobject(c_plugin.CinderClientPlugin, 'get_volume_type', side_effect=[self.vt_lvm, self.vt_ceph, self.vt_new_ceph, self.vt_ceph])
    self._set_up_qos_associate_environment()
    prop_diff = {'volume_types': [self.vt_lvm, self.vt_new_ceph]}
    self.my_qos_associate.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.qos_specs.associate.assert_any_call(self.qos_specs_id, self.vt_new_ceph)
    self.qos_specs.disassociate.assert_any_call(self.qos_specs_id, self.vt_ceph)