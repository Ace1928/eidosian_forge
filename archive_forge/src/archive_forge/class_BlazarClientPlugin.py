from blazarclient import client as blazar_client
from blazarclient import exception as client_exception
from oslo_config import cfg
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
class BlazarClientPlugin(client_plugin.ClientPlugin):
    service_types = [RESERVATION] = ['reservation']

    def _create(self, version=None):
        interface = self._get_client_option(CLIENT_NAME, 'endpoint_type')
        args = {'session': self.context.keystone_session, 'service_type': self.RESERVATION, 'interface': interface, 'region_name': self._get_region_name(), 'connect_retries': cfg.CONF.client_retry_limit}
        client = blazar_client.Client(**args)
        return client

    def is_not_found(self, exc):
        if isinstance(exc, client_exception.BlazarClientException) and exc.kwargs['code'] == 404:
            return True
        return False

    def has_host(self):
        return True if self.client().host.list() else False

    def create_lease(self, **args):
        return self.client().lease.create(**args)

    def get_lease(self, id):
        return self.client().lease.get(id)

    def create_host(self, **args):
        return self.client().host.create(**args)

    def get_host(self, id):
        return self.client().host.get(id)