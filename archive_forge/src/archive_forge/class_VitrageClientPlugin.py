from heat.engine.clients import client_plugin
from oslo_log import log as logging
from vitrageclient import client as vitrage_client
class VitrageClientPlugin(client_plugin.ClientPlugin):
    exceptions_module = None
    service_types = [RCA] = ['rca']

    def _create(self):
        return vitrage_client.Client('1', self.context.keystone_session)