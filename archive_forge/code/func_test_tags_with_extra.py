import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_tags_with_extra(self):
    self.instance_group.properties.data['Tags'] = [{'Key': 'fee', 'Value': 'foo'}]
    expected = [{'Key': 'fee', 'Value': 'foo'}, {'Value': u'asg', 'Key': 'metering.groupname'}]
    self.assertEqual(expected, self.instance_group._tags())