from unittest import mock
from heat.engine.clients.os import zaqar
from heat.tests import common
from heat.tests import utils
class ZaqarClientPluginTest(common.HeatTestCase):

    def test_create(self):
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('zaqar')
        client = plugin.client()
        self.assertIsNotNone(client.queue)

    def test_create_for_tenant(self):
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('zaqar')
        client = plugin.create_for_tenant('other_tenant', 'token')
        self.assertEqual('other_tenant', client.conf['auth_opts']['options']['os_project_id'])
        self.assertEqual('token', client.conf['auth_opts']['options']['os_auth_token'])

    def test_event_sink(self):
        context = utils.dummy_context()
        client = context.clients.client('zaqar')
        fake_queue = mock.MagicMock()
        client.queue = lambda x, auto_create: fake_queue
        sink = zaqar.ZaqarEventSink('myqueue')
        sink.consume(context, {'hello': 'world'})
        fake_queue.post.assert_called_once_with({'body': {'hello': 'world'}, 'ttl': 3600})