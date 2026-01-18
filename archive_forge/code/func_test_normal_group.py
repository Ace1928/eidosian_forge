from unittest import mock
from heat.common import grouputils
from heat.common import identifier
from heat.common import template_format
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_normal_group(self):
    group = mock.Mock()
    t = template_format.parse(nested_stack)
    stack = utils.parse_stack(t)
    group.nested.return_value = stack
    members = [r for r in stack.values()]
    expected = sorted(members, key=lambda r: (r.created_time, r.name))
    actual = grouputils.get_members(group)
    self.assertEqual(expected, actual)
    actual_ids = grouputils.get_member_refids(group)
    self.assertEqual(['ID-r0', 'ID-r1'], actual_ids)