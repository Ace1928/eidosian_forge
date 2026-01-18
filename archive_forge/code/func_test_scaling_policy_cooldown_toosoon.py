import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_scaling_policy_cooldown_toosoon(self):
    dont_call = self.patchobject(self.group, 'resize')
    self.patchobject(self.group, '_check_scaling_allowed', side_effect=resource.NoActionRequired)
    self.assertRaises(resource.NoActionRequired, self.group.adjust, 1)
    self.assertEqual([], dont_call.call_args_list)