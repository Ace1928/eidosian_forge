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
class TestScalingGroupTags(common.HeatTestCase):

    def setUp(self):
        super(TestScalingGroupTags, self).setUp()
        t = template_format.parse(as_template)
        self.stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.group = self.stack['WebServerGroup']

    def test_tags_default(self):
        expected = [{'Key': 'metering.groupname', 'Value': u'WebServerGroup'}, {'Key': 'metering.AutoScalingGroupName', 'Value': u'WebServerGroup'}]
        self.assertEqual(expected, self.group._tags())

    def test_tags_with_extra(self):
        self.group.properties.data['Tags'] = [{'Key': 'fee', 'Value': 'foo'}]
        expected = [{'Key': 'fee', 'Value': 'foo'}, {'Key': 'metering.groupname', 'Value': u'WebServerGroup'}, {'Key': 'metering.AutoScalingGroupName', 'Value': u'WebServerGroup'}]
        self.assertEqual(expected, self.group._tags())

    def test_tags_with_metering(self):
        self.group.properties.data['Tags'] = [{'Key': 'metering.fee', 'Value': 'foo'}]
        expected = [{'Key': 'metering.fee', 'Value': 'foo'}, {'Key': 'metering.AutoScalingGroupName', 'Value': u'WebServerGroup'}]
        self.assertEqual(expected, self.group._tags())