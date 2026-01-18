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
def test_volume_detach_with_error(self):
    stack_name = 'test_volume_detach_werr_stack'
    fva = self._mock_create_server_volume_script(vt_base.FakeVolume('attaching'))
    self._mock_create_volume(vt_base.FakeVolume('creating'), stack_name, mock_attachment=fva)
    self.stub_VolumeConstraint_validate()
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    self.create_volume(self.t, stack, 'DataVolume')
    rsrc = self.create_attachment(self.t, stack, 'MountPoint')
    self.fc.volumes.delete_server_volume.return_value = None
    fva = vt_base.FakeVolume('in-use')
    self.cinder_fc.volumes.get.side_effect = [vt_base.FakeVolume('error', id=fva.id)]
    detach_task = scheduler.TaskRunner(rsrc.delete)
    ex = self.assertRaises(exception.ResourceFailure, detach_task)
    self.assertIn('Volume detachment failed - Unknown status error', str(ex))
    self.fc.volumes.delete_server_volume.assert_called_once_with(u'WikiDatabase', 'vol-123')
    self.validate_mock_create_server_volume_script()