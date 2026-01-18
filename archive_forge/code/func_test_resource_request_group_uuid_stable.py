from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_resource_request_group_uuid_stable(self):
    uuid_a = place_utils.resource_request_group_uuid(namespace=self._uuid_ns, qos_rules=[mock.MagicMock(id='fake_id_0'), mock.MagicMock(id='fake_id_1')])
    uuid_b = place_utils.resource_request_group_uuid(namespace=self._uuid_ns, qos_rules=[mock.MagicMock(id='fake_id_0'), mock.MagicMock(id='fake_id_1')])
    self.assertEqual(uuid_a, uuid_b)