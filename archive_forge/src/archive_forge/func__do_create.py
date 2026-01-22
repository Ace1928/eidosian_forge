from manilaclient import api_versions
from manilaclient import base
def _do_create(self, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, share_network_id=None, metadata=None):
    """Create share network subnet.

        :param neutron_net_id: ID of Neutron network
        :param neutron_subnet_id: ID of Neutron subnet
        :param availability_zone: Name of the target availability zone
        :param metadata: dict - optional metadata to set on share creation
        :rtype: :class:`ShareNetworkSubnet`
        """
    values = {}
    if neutron_net_id:
        values['neutron_net_id'] = neutron_net_id
    if neutron_subnet_id:
        values['neutron_subnet_id'] = neutron_subnet_id
    if availability_zone:
        values['availability_zone'] = availability_zone
    if metadata:
        values['metadata'] = metadata
    body = {'share-network-subnet': values}
    url = '/share-networks/%(share_network_id)s/subnets' % {'share_network_id': share_network_id}
    return self._create(url, body, RESOURCE_NAME)