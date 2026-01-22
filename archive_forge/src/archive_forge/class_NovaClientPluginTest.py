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
class NovaClientPluginTest(NovaClientPluginTestCase):
    """Basic tests for :module:'heat.engine.clients.os.nova'."""

    def test_create(self):
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('nova')
        plugin.max_microversion = '2.53'
        client = plugin.client()
        self.assertIsNotNone(client.servers)

    def test_v2_26_create(self):
        ctxt = utils.dummy_context()
        self.patchobject(nc, 'Client', return_value=mock.Mock())
        plugin = ctxt.clients.client_plugin('nova')
        plugin.max_microversion = '2.53'
        plugin.client(version='2.26')

    def test_v2_26_create_failed(self):
        ctxt = utils.dummy_context()
        plugin = ctxt.clients.client_plugin('nova')
        plugin.max_microversion = '2.23'
        client_stub = mock.Mock()
        self.patchobject(nc, 'Client', return_value=client_stub)
        self.assertRaises(exception.InvalidServiceVersion, plugin.client, '2.26')

    def test_get_ip(self):
        my_image = mock.MagicMock()
        my_image.addresses = {'public': [{'version': 4, 'addr': '4.5.6.7'}, {'version': 6, 'addr': '2401:1801:7800:0101:c058:dd33:ff18:04e6'}], 'private': [{'version': 4, 'addr': '10.13.12.13'}]}
        expected = '4.5.6.7'
        observed = self.nova_plugin.get_ip(my_image, 'public', 4)
        self.assertEqual(expected, observed)
        expected = '10.13.12.13'
        observed = self.nova_plugin.get_ip(my_image, 'private', 4)
        self.assertEqual(expected, observed)
        expected = '2401:1801:7800:0101:c058:dd33:ff18:04e6'
        observed = self.nova_plugin.get_ip(my_image, 'public', 6)
        self.assertEqual(expected, observed)

    def test_find_flavor_by_name_or_id(self):
        """Tests the find_flavor_by_name_or_id function."""
        flav_id = str(uuid.uuid4())
        flav_name = 'X-Large'
        my_flavor = mock.MagicMock()
        my_flavor.name = flav_name
        my_flavor.id = flav_id
        self.nova_client.flavors.get.side_effect = [my_flavor, nova_exceptions.NotFound(''), nova_exceptions.NotFound('')]
        self.nova_client.flavors.find.side_effect = [my_flavor, nova_exceptions.NotFound('')]
        self.assertEqual(flav_id, self.nova_plugin.find_flavor_by_name_or_id(flav_id))
        self.assertEqual(flav_id, self.nova_plugin.find_flavor_by_name_or_id(flav_name))
        self.assertRaises(nova_exceptions.ClientException, self.nova_plugin.find_flavor_by_name_or_id, 'noflavor')
        self.assertEqual(3, self.nova_client.flavors.get.call_count)
        self.assertEqual(2, self.nova_client.flavors.find.call_count)

    def test_get_host(self):
        """Tests the get_host function."""
        my_hypervisor_hostname = 'myhost'
        my_host = mock.MagicMock()
        my_host.hypervisor_hostname = my_hypervisor_hostname
        self.nova_client.hypervisors.search.side_effect = [my_host, nova_exceptions.NotFound(404)]
        self.assertEqual(my_host, self.nova_plugin.get_host(my_hypervisor_hostname))
        self.assertRaises(nova_exceptions.NotFound, self.nova_plugin.get_host, 'nohost')
        self.assertEqual(2, self.nova_client.hypervisors.search.call_count)
        calls = [mock.call('myhost'), mock.call('nohost')]
        self.assertEqual(calls, self.nova_client.hypervisors.search.call_args_list)

    def test_get_keypair(self):
        """Tests the get_keypair function."""
        my_pub_key = 'a cool public key string'
        my_key_name = 'mykey'
        my_key = mock.MagicMock()
        my_key.public_key = my_pub_key
        my_key.name = my_key_name
        self.nova_client.keypairs.get.side_effect = [my_key, nova_exceptions.NotFound(404)]
        self.assertEqual(my_key, self.nova_plugin.get_keypair(my_key_name))
        self.assertRaises(exception.EntityNotFound, self.nova_plugin.get_keypair, 'notakey')
        calls = [mock.call(my_key_name), mock.call('notakey')]
        self.nova_client.keypairs.get.assert_has_calls(calls)

    def test_get_server(self):
        """Tests the get_server function."""
        my_server = mock.MagicMock()
        self.nova_client.servers.get.side_effect = [my_server, nova_exceptions.NotFound(404)]
        self.assertEqual(my_server, self.nova_plugin.get_server('my_server'))
        self.assertRaises(exception.EntityNotFound, self.nova_plugin.get_server, 'idontexist')
        calls = [mock.call('my_server'), mock.call('idontexist')]
        self.nova_client.servers.get.assert_has_calls(calls)

    def test_get_status(self):
        server = mock.Mock()
        server.status = 'ACTIVE'
        observed = self.nova_plugin.get_status(server)
        self.assertEqual('ACTIVE', observed)
        server.status = 'ACTIVE(STATUS)'
        observed = self.nova_plugin.get_status(server)
        self.assertEqual('ACTIVE', observed)

    def test_check_verify_resize_task_state(self):
        """Tests the check_verify_resize function with resize task_state."""
        my_server = mock.MagicMock(status='Foo')
        setattr(my_server, 'OS-EXT-STS:task_state', 'resize_finish')
        self.nova_client.servers.get.side_effect = [my_server]
        self.assertEqual(False, self.nova_plugin.check_verify_resize('my_server'))

    def test_check_verify_resize_error(self):
        """Tests the check_verify_resize function with unknown status."""
        my_server = mock.MagicMock(status='Foo')
        setattr(my_server, 'OS-EXT-STS:task_state', 'active')
        self.nova_client.servers.get.side_effect = [my_server]
        self.assertRaises(exception.ResourceUnknownStatus, self.nova_plugin.check_verify_resize, 'my_server')

    def _absolute_limits(self):
        max_personality = mock.Mock()
        max_personality.name = 'maxPersonality'
        max_personality.value = 5
        max_personality_size = mock.Mock()
        max_personality_size.name = 'maxPersonalitySize'
        max_personality_size.value = 10240
        max_server_meta = mock.Mock()
        max_server_meta.name = 'maxServerMeta'
        max_server_meta.value = 3
        yield max_personality
        yield max_personality_size
        yield max_server_meta

    def test_absolute_limits_success(self):
        limits = mock.Mock()
        limits.absolute = self._absolute_limits()
        self.nova_client.limits.get.return_value = limits
        self.nova_plugin.absolute_limits()

    def test_absolute_limits_retry(self):
        limits = mock.Mock()
        limits.absolute = self._absolute_limits()
        self.nova_client.limits.get.side_effect = [requests.ConnectionError, requests.ConnectionError, limits]
        self.nova_plugin.absolute_limits()
        self.assertEqual(3, self.nova_client.limits.get.call_count)

    def test_absolute_limits_failure(self):
        limits = mock.Mock()
        limits.absolute = self._absolute_limits()
        self.nova_client.limits.get.side_effect = [requests.ConnectionError, requests.ConnectionError, requests.ConnectionError]
        self.assertRaises(requests.ConnectionError, self.nova_plugin.absolute_limits)