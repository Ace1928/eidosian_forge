import base64
import copy
import datetime
import json
import time
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import crypt
from oauth2client import transport
def _create_token(self, additional_claims=None):
    now = client._UTCNOW()
    lifetime = datetime.timedelta(seconds=self._MAX_TOKEN_LIFETIME_SECS)
    expiry = now + lifetime
    payload = {'iat': _datetime_to_secs(now), 'exp': _datetime_to_secs(expiry), 'iss': self._service_account_email, 'sub': self._service_account_email}
    payload.update(self._kwargs)
    if additional_claims is not None:
        payload.update(additional_claims)
    jwt = crypt.make_signed_jwt(self._signer, payload, key_id=self._private_key_id)
    return (jwt.decode('ascii'), expiry)