import collections
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_volume_type_get_live_state_public(self):
    self.my_volume_type.resource_id = '1234'
    volume_type = mock.Mock()
    volume_type.to_dict.return_value = {'name': 'test', 'is_public': True, 'description': 'test1', 'metadata': {'one': 'two'}}
    self.volume_types.get.return_value = volume_type
    volume_type.get_keys.return_value = {'one': 'two'}
    volume_type_access = mock.MagicMock()
    self.cinderclient.volume_type_access = volume_type_access
    reality = self.my_volume_type.get_live_state(self.my_volume_type.properties)
    expected = {'name': 'test', 'is_public': True, 'description': 'test1', 'projects': [], 'metadata': {'one': 'two'}}
    self.assertEqual(set(expected.keys()), set(reality.keys()))
    for key in reality:
        self.assertEqual(expected[key], reality[key])
    self.assertEqual(0, volume_type_access.list.call_count)