import collections
import copy
import json
from unittest import mock
from cinderclient import exceptions as cinder_exp
from novaclient import exceptions as nova_exp
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.resources.openstack.cinder import volume as c_vol
from heat.engine.resources import scheduler_hints as sh
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests.openstack.cinder import test_volume_utils as vt_base
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _update_if_attached(self, stack_name, update_type='resize'):
    self.stub_VolumeConstraint_validate()
    fv3 = vt_base.FakeVolume('attaching')
    fv3_ready = vt_base.FakeVolume('in-use', id=fv3.id)
    fv1 = self._mock_create_server_volume_script(vt_base.FakeVolume('attaching'), extra_create_server_volume_mocks=[fv3])
    attachments = [{'id': 'vol-123', 'device': '/dev/vdc', 'server_id': u'WikiDatabase'}]
    fv2 = vt_base.FakeVolume('in-use', attachments=attachments, size=1)
    fvd = vt_base.FakeVolume('in-use')
    resize_m_get = [vt_base.FakeVolume('extending'), vt_base.FakeVolume('extending'), vt_base.FakeVolume('available')]
    extra_get_mocks = [fv1, fv2, fvd, vt_base.FakeVolume('available')]
    if update_type == 'resize':
        extra_get_mocks += resize_m_get
    extra_get_mocks.append(fv3_ready)
    self._mock_create_volume(vt_base.FakeVolume('creating'), stack_name, extra_get_mocks=extra_get_mocks)
    self.fc.volumes.get_server_volume.side_effect = [fvd, fvd, fakes_nova.fake_exception()]
    if update_type == 'access_mode':
        update_readonly_mock = self.patchobject(self.cinder_fc.volumes, 'update_readonly_flag', return_value=None)
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    rsrc = self.create_volume(self.t, stack, 'volume')
    self.create_attachment(self.t, stack, 'attachment')
    props = copy.deepcopy(rsrc.properties.data)
    if update_type == 'access_mode':
        props['read_only'] = True
    if update_type == 'resize':
        props['size'] = 2
    after = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    update_task = scheduler.TaskRunner(rsrc.update, after)
    self.assertIsNone(update_task())
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    if update_type == 'access_mode':
        update_readonly_mock.assert_called_once_with(fvd.id, True)
    if update_type == 'resize':
        self.cinder_fc.volumes.extend.assert_called_once_with(fvd.id, 2)
    self.fc.volumes.get_server_volume.assert_called_with(u'WikiDatabase', 'vol-123')
    self.fc.volumes.delete_server_volume.assert_called_with('WikiDatabase', 'vol-123')