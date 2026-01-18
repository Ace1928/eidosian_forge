import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def update_region(self, region_id, region_ref):
    """Update region by id.

        :returns: region_ref dict
        :raises keystone.exception.RegionNotFound: If the region doesn't exist.

        """
    raise exception.NotImplemented()