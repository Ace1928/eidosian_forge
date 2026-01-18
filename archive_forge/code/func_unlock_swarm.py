import logging
import http.client as http_client
from ..constants import DEFAULT_SWARM_ADDR_POOL, DEFAULT_SWARM_SUBNET_SIZE
from .. import errors
from .. import types
from .. import utils
@utils.minimum_version('1.24')
def unlock_swarm(self, key):
    """
            Unlock a locked swarm.

            Args:
                key (string): The unlock key as provided by
                    :py:meth:`get_unlock_key`

            Raises:
                :py:class:`docker.errors.InvalidArgument`
                    If the key argument is in an incompatible format

                :py:class:`docker.errors.APIError`
                    If the server returns an error.

            Returns:
                `True` if the request was successful.

            Example:

                >>> key = client.api.get_unlock_key()
                >>> client.unlock_swarm(key)

        """
    if isinstance(key, dict):
        if 'UnlockKey' not in key:
            raise errors.InvalidArgument('Invalid unlock key format')
    else:
        key = {'UnlockKey': key}
    url = self._url('/swarm/unlock')
    res = self._post_json(url, data=key)
    self._raise_for_status(res)
    return True