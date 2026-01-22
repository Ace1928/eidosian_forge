import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class NovaClientPluginMetadataTest(NovaClientPluginTestCase):

    def test_serialize_string(self):
        original = {'test_key': 'simple string value'}
        self.assertEqual(original, self.nova_plugin.meta_serialize(original))

    def test_serialize_int(self):
        original = {'test_key': 123}
        expected = {'test_key': '123'}
        self.assertEqual(expected, self.nova_plugin.meta_serialize(original))

    def test_serialize_list(self):
        original = {'test_key': [1, 2, 3]}
        expected = {'test_key': '[1, 2, 3]'}
        self.assertEqual(expected, self.nova_plugin.meta_serialize(original))

    def test_serialize_dict(self):
        original = collections.OrderedDict([('test_key', collections.OrderedDict([('a', 'b'), ('c', 'd')]))])
        expected = {'test_key': '{"a": "b", "c": "d"}'}
        actual = self.nova_plugin.meta_serialize(original)
        self.assertEqual(json.loads(expected['test_key']), json.loads(actual['test_key']))

    def test_serialize_none(self):
        original = {'test_key': None}
        expected = {'test_key': 'null'}
        self.assertEqual(expected, self.nova_plugin.meta_serialize(original))

    def test_serialize_no_value(self):
        """Prove that the user can only pass in a dict to nova metadata."""
        excp = self.assertRaises(exception.StackValidationFailed, self.nova_plugin.meta_serialize, 'foo')
        self.assertIn('metadata needs to be a Map', str(excp))

    def test_serialize_combined(self):
        original = {'test_key_1': 123, 'test_key_2': 'a string', 'test_key_3': {'a': 'b'}, 'test_key_4': None}
        expected = {'test_key_1': '123', 'test_key_2': 'a string', 'test_key_3': '{"a": "b"}', 'test_key_4': 'null'}
        self.assertEqual(expected, self.nova_plugin.meta_serialize(original))