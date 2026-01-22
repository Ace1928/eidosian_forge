from openstack import exceptions
from openstack import resource
from openstack import utils
def add_bgp_peer(self, session, peer_id):
    """Add BGP Peer to a BGP Speaker

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param peer_id: id of the peer to associate with the speaker.

        :returns: A dictionary as the API Reference describes it.

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
    url = utils.urljoin(self.base_path, self.id, 'add_bgp_peer')
    body = {'bgp_peer_id': peer_id}
    resp = self._put(session, url, body)
    return resp.json()