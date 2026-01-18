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
def test_segment_delete_resource_id_is_none(self):
    self.segment.resource_id = None
    mock_delete = self.patchobject(self.sdkclient.network, 'delete_segment')
    self.assertIsNone(self.segment.handle_delete())
    self.assertEqual(0, mock_delete.call_count)