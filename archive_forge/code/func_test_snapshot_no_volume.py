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
def test_snapshot_no_volume(self):
    """Test that backup does not start for failed resource."""
    stack_name = 'test_volume_snapshot_novol_stack'
    cfg.CONF.set_override('action_retry_limit', 0)
    fva = vt_base.FakeVolume('error')
    fv = self._mock_create_volume(vt_base.FakeVolume('creating'), stack_name, final_status='error', mock_attachment=fva)
    self._mock_delete_volume(fv)
    self.t['Resources']['DataVolume']['DeletionPolicy'] = 'Snapshot'
    self.t['Resources']['DataVolume']['Properties']['AvailabilityZone'] = 'nova'
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    resource_defns = stack.t.resource_definitions(stack)
    rsrc = aws_vol.Volume('DataVolume', resource_defns['DataVolume'], stack)
    create = scheduler.TaskRunner(rsrc.create)
    ex = self.assertRaises(exception.ResourceFailure, create)
    self.assertIn('Went to status error due to "Unknown"', str(ex))
    self.cinder_fc.volumes.get.side_effect = [fva, cinder_exp.NotFound('Not found')]
    scheduler.TaskRunner(rsrc.destroy)()