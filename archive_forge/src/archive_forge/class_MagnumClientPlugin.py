from magnumclient import exceptions as mc_exc
from magnumclient.v1 import client as magnum_client
from oslo_config import cfg
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
class MagnumClientPlugin(client_plugin.ClientPlugin):
    service_types = [CONTAINER] = ['container-infra']

    def _create(self):
        interface = self._get_client_option(CLIENT_NAME, 'endpoint_type')
        args = {'interface': interface, 'service_type': self.CONTAINER, 'session': self.context.keystone_session, 'connect_retries': cfg.CONF.client_retry_limit, 'region_name': self._get_region_name()}
        client = magnum_client.Client(**args)
        return client

    def is_not_found(self, ex):
        return isinstance(ex, mc_exc.NotFound)

    def is_over_limit(self, ex):
        return isinstance(ex, mc_exc.RequestEntityTooLarge)

    def is_conflict(self, ex):
        return isinstance(ex, mc_exc.Conflict)

    def _get_rsrc_name_or_id(self, value, entity, entity_msg):
        entity_client = getattr(self.client(), entity)
        try:
            return entity_client.get(value).uuid
        except mc_exc.NotFound:
            raise exception.EntityNotFound(entity=entity_msg, name=value)

    def get_cluster_template(self, value):
        return self._get_rsrc_name_or_id(value, entity='cluster_templates', entity_msg='ClusterTemplate')