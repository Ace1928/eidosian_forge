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
def test_segment_handle_delete_not_found(self):
    segment_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    self.segment.resource_id = segment_id
    mock_delete = self.patchobject(self.sdkclient.network, 'delete_segment', side_effect=exceptions.ResourceNotFound)
    self.assertIsNone(self.segment.handle_delete())
    mock_delete.assert_called_once_with(self.segment.resource_id)