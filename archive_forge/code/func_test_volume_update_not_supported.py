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
def test_volume_update_not_supported(self):
    stack_name = 'test_volume_updnotsup_stack'
    fv = vt_base.FakeVolume('creating')
    self._mock_create_volume(fv, stack_name)
    t = template_format.parse(volume_template)
    stack = utils.parse_stack(t, stack_name=stack_name)
    rsrc = self.create_volume(t, stack, 'DataVolume')
    props = copy.deepcopy(rsrc.properties.data)
    props['Size'] = 2
    props['Tags'] = None
    props['AvailabilityZone'] = 'other'
    after = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    updater = scheduler.TaskRunner(rsrc.update, after)
    ex = self.assertRaises(exception.ResourceFailure, updater)
    self.assertIn('NotSupported: resources.DataVolume: Update to properties AvailabilityZone, Size, Tags of DataVolume (AWS::EC2::Volume) is not supported', str(ex))
    self.assertEqual((rsrc.UPDATE, rsrc.FAILED), rsrc.state)