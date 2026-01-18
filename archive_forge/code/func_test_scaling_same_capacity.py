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
def test_scaling_same_capacity(self):
    """Don't resize when capacity is the same."""
    self.patchobject(grouputils, 'get_size', return_value=3)
    resize = self.patchobject(self.group, 'resize')
    finished_scaling = self.patchobject(self.group, '_finished_scaling')
    notify = self.patch('heat.engine.notification.autoscaling.send')
    self.assertRaises(resource.NoActionRequired, self.group.adjust, 3, adjustment_type='ExactCapacity')
    expected_notifies = []
    self.assertEqual(expected_notifies, notify.call_args_list)
    self.assertEqual(0, resize.call_count)
    self.assertEqual(0, finished_scaling.call_count)