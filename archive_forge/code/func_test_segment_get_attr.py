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
def test_segment_get_attr(self):
    segment_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    self.segment.resource_id = segment_id
    seg = {'name': 'test_segment', 'id': '477e8273-60a7-4c41-b683-fdb0bc7cd151', 'network_type': 'vxlan', 'network_id': 'private', 'segmentation_id': 101}

    class FakeSegment(object):

        def to_dict(self):
            return seg
    get_mock = self.patchobject(self.sdkclient.network, 'get_segment', return_value=FakeSegment())
    self.assertEqual(seg, self.segment.FnGetAtt('show'))
    get_mock.assert_called_once_with(self.segment.resource_id)