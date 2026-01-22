import abc
from keystone.common import manager
import keystone.conf
from keystone import exception
class IDGenerator(object, metaclass=abc.ABCMeta):
    """Interface description for an ID Generator provider."""

    @abc.abstractmethod
    def generate_public_ID(self, mapping):
        """Return a Public ID for the given mapping dict.

        :param dict mapping: The items to be hashed.

        The ID must be reproducible and no more than 64 chars in length.
        The ID generated should be independent of the order of the items
        in the mapping dict.

        """
        raise exception.NotImplemented()