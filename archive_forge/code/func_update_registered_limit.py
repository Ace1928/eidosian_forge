import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def update_registered_limit(self, registered_limit_id, registered_limit):
    """Update existing registered limits.

        :param registered_limit_id: the id of the registered limit.
        :param registered_limit: a dict containing the registered limit
                                 attributes to update.
        :returns: the updated registered limit.
        :raises keystone.exception.RegisteredLimitNotFound: If registered limit
            doesn't exist.
        :raises keystone.exception.Conflict: If update to a duplicate
            registered limit.

        """
    raise exception.NotImplemented()