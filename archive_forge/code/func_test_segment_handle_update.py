from unittest import mock
from neutronclient.neutron import v2_0 as neutronV20
from openstack import exceptions
from oslo_utils import excutils
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def test_segment_handle_update(self):
    segment_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    self.segment.resource_id = segment_id
    props = {'name': 'test_segment', 'description': 'updated'}
    mock_update = self.patchobject(self.sdkclient.network, 'update_segment')
    update_dict = props.copy()
    update_snippet = rsrc_defn.ResourceDefinition(self.segment.name, self.segment.type(), props)
    self.segment.handle_update(json_snippet=update_snippet, tmpl_diff={}, prop_diff=props)
    props['name'] = None
    self.segment.handle_update(json_snippet=update_snippet, tmpl_diff={}, prop_diff=props)
    self.assertEqual(2, mock_update.call_count)
    mock_update.assert_called_with(segment_id, **update_dict)