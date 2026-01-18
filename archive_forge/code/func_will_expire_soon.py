import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
def will_expire_soon(self, stale_duration=STALE_TOKEN_DURATION):
    """Determine if expiration is about to occur.

        :returns: true if expiration is within the given duration
        :rtype: boolean

        """
    norm_expires = utils.normalize_time(self.expires)
    soon = utils.from_utcnow(seconds=stale_duration)
    return norm_expires < soon