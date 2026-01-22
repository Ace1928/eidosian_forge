from openstack import exceptions
from openstack import resource
from openstack import utils
def add_gateway_network(self, session, network_id):
    """Add Network to a BGP Speaker

        :param: session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param network_id: The ID of the network to associate with the speaker

        :returns: A dictionary as the API Reference describes it.
        """
    body = {'network_id': network_id}
    url = utils.urljoin(self.base_path, self.id, 'add_gateway_network')
    resp = session.put(url, json=body)
    return resp.json()