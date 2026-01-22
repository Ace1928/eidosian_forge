from heat.common import exception as heat_exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
from manilaclient import client as manila_client
from manilaclient import exceptions
from oslo_config import cfg
class ManilaClientPlugin(client_plugin.ClientPlugin):
    exceptions_module = exceptions
    service_types = [SHARE] = ['sharev2']

    def _create(self):
        endpoint_type = self._get_client_option(CLIENT_NAME, 'endpoint_type')
        args = {'endpoint_type': endpoint_type, 'service_type': self.SHARE, 'session': self.context.keystone_session, 'connect_retries': cfg.CONF.client_retry_limit, 'region_name': self._get_region_name()}
        client = manila_client.Client(MANILACLIENT_VERSION, **args)
        return client

    def is_not_found(self, ex):
        return isinstance(ex, exceptions.NotFound)

    def is_over_limit(self, ex):
        return isinstance(ex, exceptions.RequestEntityTooLarge)

    def is_conflict(self, ex):
        return isinstance(ex, exceptions.Conflict)

    @staticmethod
    def _find_resource_by_id_or_name(id_or_name, resource_list, resource_type_name):
        """The method is trying to find id or name in item_list

        The method searches item with id_or_name in list and returns it.
        If there is more than one value or no values then it raises an
        exception

        :param id_or_name: resource id or name
        :param resource_list: list of resources
        :param resource_type_name: name of resource type that will be used
                                   for exceptions
        :raises EntityNotFound: if cannot find resource by name
        :raises NoUniqueMatch: if find more than one resource by ambiguous name
        :return: resource or generate an exception otherwise
        """
        search_result_by_id = [res for res in resource_list if res.id == id_or_name]
        if search_result_by_id:
            return search_result_by_id[0]
        else:
            search_result_by_name = [res for res in resource_list if res.name == id_or_name]
            match_count = len(search_result_by_name)
            if match_count > 1:
                message = "Ambiguous {0} name '{1}'. Found more than one {0} for this name in Manila.".format(resource_type_name, id_or_name)
                raise exceptions.NoUniqueMatch(message)
            elif match_count == 1:
                return search_result_by_name[0]
            else:
                raise heat_exception.EntityNotFound(entity=resource_type_name, name=id_or_name)

    def get_share_type(self, share_type_identity):
        return self._find_resource_by_id_or_name(share_type_identity, self.client().share_types.list(), 'share type')

    def get_share_network(self, share_network_identity):
        return self._find_resource_by_id_or_name(share_network_identity, self.client().share_networks.list(), 'share network')

    def get_share_snapshot(self, snapshot_identity):
        return self._find_resource_by_id_or_name(snapshot_identity, self.client().share_snapshots.list(), 'share snapshot')

    def get_security_service(self, service_identity):
        return self._find_resource_by_id_or_name(service_identity, self.client().security_services.list(), 'security service')