from openstack import exceptions
from openstack import resource
from openstack import utils
def add_addresses(self, session, addresses):
    """Add addresses into the address group.

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param list addresses: The list of address strings.

        :returns: The response as a AddressGroup object with updated addresses

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
    url = utils.urljoin(self.base_path, self.id, 'add_addresses')
    resp = self._put(session, url, {'addresses': addresses})
    self._translate_response(resp)
    return self