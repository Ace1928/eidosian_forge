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
def _make_jwt_for_audience(self, audience):
    """Make a new JWT for the given audience.

        Args:
            audience (str): The intended audience.

        Returns:
            Tuple[bytes, datetime]: The encoded JWT and the expiration.
        """
    now = _helpers.utcnow()
    lifetime = datetime.timedelta(seconds=self._token_lifetime)
    expiry = now + lifetime
    payload = {'iss': self._issuer, 'sub': self._subject, 'iat': _helpers.datetime_to_secs(now), 'exp': _helpers.datetime_to_secs(expiry), 'aud': audience}
    payload.update(self._additional_claims)
    jwt = encode(self._signer, payload)
    return (jwt, expiry)