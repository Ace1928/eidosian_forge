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
class NovaClientPluginFetchServerTest(NovaClientPluginTestCase):
    server = mock.Mock()
    server.id = '1234'
    server.name = 'test_fetch_server'
    msg = 'ClientException: The server has either erred or is incapable of performing the requested operation.'
    scenarios = [('successful_refresh', dict(value=server, e_raise=False)), ('overlimit_error', dict(value=nova_exceptions.OverLimit(413, 'limit reached'), e_raise=False)), ('500_error', dict(value=nova_exceptions.ClientException(500, msg), e_raise=False)), ('503_error', dict(value=nova_exceptions.ClientException(503, msg), e_raise=False)), ('unhandled_exception', dict(value=nova_exceptions.ClientException(501, msg), e_raise=True))]

    def test_fetch_server(self):
        self.nova_client.servers.get.side_effect = [self.value]
        if self.e_raise:
            self.assertRaises(nova_exceptions.ClientException, self.nova_plugin.fetch_server, self.server.id)
        elif isinstance(self.value, mock.Mock):
            self.assertEqual(self.value, self.nova_plugin.fetch_server(self.server.id))
        else:
            self.assertIsNone(self.nova_plugin.fetch_server(self.server.id))
        self.nova_client.servers.get.assert_called_once_with(self.server.id)