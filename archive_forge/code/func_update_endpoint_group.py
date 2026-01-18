import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def update_endpoint_group(self, endpoint_group_id, endpoint_group):
    """Update an endpoint group.

        :param endpoint_group_id: identity of endpoint group to retrieve
        :type endpoint_group_id: string
        :param endpoint_group: A full or partial endpoint_group
        :type endpoint_group: dictionary
        :raises keystone.exception.NotFound: If the endpoint group was not
            found.
        :returns: an endpoint group representation.

        """
    raise exception.NotImplemented()