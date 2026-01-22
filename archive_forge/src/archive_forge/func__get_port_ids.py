from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def _get_port_ids(self, name_or_id_list, filters=None):
    """
        Takes a list of port names or ids, retrieves ports and returns a list
        with port ids only.

        :param list[str] name_or_id_list: list of port names or ids
        :param dict filters: optional filters
        :raises: SDKException on multiple matches
        :raises: ResourceNotFound if a port is not found
        :return: list of port ids
        :rtype: list[str]
        """
    ids_list = []
    for name_or_id in name_or_id_list:
        port = self.get_port(name_or_id, filters)
        if not port:
            raise exceptions.ResourceNotFound('Port {id} not found'.format(id=name_or_id))
        ids_list.append(port['id'])
    return ids_list