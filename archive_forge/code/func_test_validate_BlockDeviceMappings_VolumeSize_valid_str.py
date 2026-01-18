import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_validate_BlockDeviceMappings_VolumeSize_valid_str(self):
    stack_name = 'val_VolumeSize_valid'
    tmpl, stack = self._setup_test_stack(stack_name)
    bdm = [{'DeviceName': 'vdb', 'Ebs': {'SnapshotId': '1234', 'VolumeSize': '1'}}]
    wsp = tmpl.t['Resources']['WebServer']['Properties']
    wsp['BlockDeviceMappings'] = bdm
    resource_defns = tmpl.resource_definitions(stack)
    instance = instances.Instance('validate_volume_size', resource_defns['WebServer'], stack)
    self._mock_get_image_id_success('F17-x86_64-gold', 1)
    self.stub_SnapshotConstraint_validate()
    self.stub_VolumeConstraint_validate()
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    self.assertIsNone(instance.validate())