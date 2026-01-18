from heat.common import exception as heat_exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
from manilaclient import client as manila_client
from manilaclient import exceptions
from oslo_config import cfg
The method is trying to find id or name in item_list

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
        