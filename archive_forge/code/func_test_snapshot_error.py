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
def test_snapshot_error(self):
    stack_name = 'test_volume_snapshot_err_stack'
    fv = self._mock_create_volume(vt_base.FakeVolume('creating'), stack_name)
    fb = vt_base.FakeBackup('error')
    self.m_backups.create.return_value = fb
    self.m_backups.get.return_value = fb
    self.t['Resources']['DataVolume']['DeletionPolicy'] = 'Snapshot'
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    rsrc = self.create_volume(self.t, stack, 'DataVolume')
    ex = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.destroy))
    self.assertIn('Unknown status error', str(ex))
    self.m_backups.create.assert_called_once_with(fv.id)
    self.m_backups.get.assert_called_once_with(fb.id)