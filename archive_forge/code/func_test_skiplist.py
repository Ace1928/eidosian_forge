import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_skiplist(self):
    stack = utils.parse_stack(template)
    resg = stack['group1']
    if self.data_in is not None:
        resg.resource_id = 'foo'
    properties = mock.MagicMock()
    p_data = {'removal_policies': [{'resource_list': self.rm_list}], 'removal_policies_mode': self.rm_mode}
    properties.get.side_effect = p_data.get
    resg.data = mock.Mock()
    resg.data.return_value.get.return_value = self.data_in
    resg.data_set = mock.Mock()
    mock_inspect = mock.Mock()
    self.patchobject(grouputils.GroupInspector, 'from_parent_resource', return_value=mock_inspect)
    mock_inspect.member_names.return_value = self.nested_rsrcs
    if not self.fallback:
        refs_map = {n: 'id-%s' % n for n in self.nested_rsrcs}
        resg.get_output = mock.Mock(return_value=refs_map)
    else:
        resg.get_output = mock.Mock(side_effect=exception.NotFound)

        def stack_contains(name):
            return name in self.nested_rsrcs

        def by_refid(name):
            rid = name.replace('id-', '')
            if rid not in self.nested_rsrcs:
                return None
            res = mock.Mock()
            res.name = rid
            return res
        nested = mock.MagicMock()
        nested.__contains__.side_effect = stack_contains
        nested.__iter__.side_effect = iter(self.nested_rsrcs)
        nested.resource_by_refid.side_effect = by_refid
        resg.nested = mock.Mock(return_value=nested)
    resg._update_name_skiplist(properties)
    if self.saved:
        resg.data_set.assert_called_once_with('name_blacklist', ','.join(self.expected))
    else:
        resg.data_set.assert_not_called()
        self.assertEqual(set(self.expected), resg._name_skiplist())