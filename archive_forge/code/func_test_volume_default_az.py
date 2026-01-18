import copy
from unittest import mock
from cinderclient import exceptions as cinder_exp
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import nova
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.aws.ec2 import volume as aws_vol
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.openstack.cinder import test_volume_utils as vt_base
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_volume_default_az(self):
    fv = vt_base.FakeVolume('creating')
    stack_name = 'test_volume_defaultaz_stack'
    self.patchobject(instance.Instance, 'handle_create')
    self.patchobject(instance.Instance, 'check_create_complete', return_value=True)
    self.patchobject(instance.Instance, '_resolve_attribute', return_value=None)
    self.patchobject(aws_vol.VolumeAttachment, 'handle_create')
    self.patchobject(aws_vol.VolumeAttachment, 'check_create_complete', return_value=True)
    cinder.CinderClientPlugin._create.return_value = self.cinder_fc
    self.stub_ImageConstraint_validate()
    self.stub_ServerConstraint_validate()
    self.stub_VolumeConstraint_validate()
    vol_name = utils.PhysName(stack_name, 'DataVolume')
    self.cinder_fc.volumes.create.return_value = fv
    fv_ready = vt_base.FakeVolume('available', id=fv.id)
    self.cinder_fc.volumes.get.side_effect = [fv, fv_ready, cinder_exp.NotFound('Not found')]
    cookie = object()
    self.patchobject(instance.Instance, 'handle_delete')
    self.patchobject(aws_vol.VolumeAttachment, 'handle_delete', return_value=cookie)
    self.patchobject(aws_vol.VolumeAttachment, 'check_delete_complete', return_value=True)
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    stack._update_all_resource_data(True, False)
    rsrc = stack['DataVolume']
    self.assertIsNone(rsrc.validate())
    scheduler.TaskRunner(stack.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    scheduler.TaskRunner(stack.delete)()
    instance.Instance._resolve_attribute.assert_called_with('AvailabilityZone')
    self.cinder_fc.volumes.create.assert_called_once_with(size=1, availability_zone=None, description=vol_name, name=vol_name, metadata={u'Usage': u'Wiki Data Volume'})
    self.cinder_fc.volumes.get.assert_called_with('vol-123')
    aws_vol.VolumeAttachment.check_delete_complete.assert_called_once_with(cookie)