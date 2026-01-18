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
def test_scaling_policy_resize_fail(self):
    self.patchobject(grouputils, 'get_size', return_value=0)
    self.patchobject(self.group, 'resize', side_effect=ValueError('test error'))
    notify = self.patch('heat.engine.notification.autoscaling.send')
    self.patchobject(self.group, '_check_scaling_allowed')
    self.patchobject(self.group, '_finished_scaling')
    self.assertRaises(ValueError, self.group.adjust, 1)
    expected_notifies = [mock.call(capacity=0, suffix='start', adjustment_type='ChangeInCapacity', groupname=u'WebServerGroup', message=u'Start resizing the group WebServerGroup', adjustment=1, stack=self.group.stack), mock.call(capacity=0, suffix='error', adjustment_type='ChangeInCapacity', groupname=u'WebServerGroup', message=u'test error', adjustment=1, stack=self.group.stack)]
    self.assertEqual(expected_notifies, notify.call_args_list)
    grouputils.get_size.assert_called_with(self.group)