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
def test_volume_attachment_error(self):
    stack_name = 'test_volume_attach_error_stack'
    mock_attachment = self._mock_create_server_volume_script(vt_base.FakeVolume('attaching'), final_status='error')
    self._mock_create_volume(vt_base.FakeVolume('creating'), stack_name, mock_attachment=mock_attachment)
    self.stub_VolumeConstraint_validate()
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    self.create_volume(self.t, stack, 'DataVolume')
    ex = self.assertRaises(exception.ResourceFailure, self.create_attachment, self.t, stack, 'MountPoint')
    self.assertIn('Volume attachment failed - Unknown status error', str(ex))
    self.validate_mock_create_server_volume_script()