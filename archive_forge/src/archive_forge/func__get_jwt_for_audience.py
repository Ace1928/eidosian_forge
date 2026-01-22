import copy
import datetime
import json
import cachetools
import six
from six.moves import urllib
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import crypt
from google.auth import exceptions
import google.auth.credentials
def _get_jwt_for_audience(self, audience):
    """Get a JWT For a given audience.

        If there is already an existing, non-expired token in the cache for
        the audience, that token is used. Otherwise, a new token will be
        created.

        Args:
            audience (str): The intended audience.

        Returns:
            bytes: The encoded JWT.
        """
    token, expiry = self._cache.get(audience, (None, None))
    if token is None or expiry < _helpers.utcnow():
        token, expiry = self._make_jwt_for_audience(audience)
        self._cache[audience] = (token, expiry)
    return token