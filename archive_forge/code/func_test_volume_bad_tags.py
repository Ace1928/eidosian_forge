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
def test_volume_bad_tags(self):
    stack_name = 'test_volume_bad_tags_stack'
    self.t['Resources']['DataVolume']['Properties']['Tags'] = [{'Foo': 'bar'}]
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    ex = self.assertRaises(exception.StackValidationFailed, self.create_volume, self.t, stack, 'DataVolume')
    self.assertEqual('Property error: Resources.DataVolume.Properties.Tags[0]: Unknown Property Foo', str(ex))