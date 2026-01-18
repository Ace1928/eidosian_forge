import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_output_attribute_dict(self):
    mock_members = self.patchobject(grouputils, 'get_members')
    members = []
    output = {}
    for ip_ex in range(1, 4):
        inst = mock.Mock()
        inst.name = str(ip_ex)
        inst.FnGetAtt.return_value = '2.1.3.%d' % ip_ex
        output[str(ip_ex)] = '2.1.3.%d' % ip_ex
        members.append(inst)
    mock_members.return_value = members
    self.assertEqual(output, self.group.FnGetAtt('outputs', 'Bar'))