import datetime
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import credentials
from google.auth import exceptions
from google.auth import jwt
from google.oauth2 import _client
def _create_jwt(self):
    now = _helpers.utcnow()
    expiry = now + JWT_LIFETIME
    iss_sub_value = 'system:serviceaccount:{}:{}'.format(self._project, self._service_identity_name)
    payload = {'iss': iss_sub_value, 'sub': iss_sub_value, 'aud': self._token_uri, 'iat': _helpers.datetime_to_secs(now), 'exp': _helpers.datetime_to_secs(expiry)}
    return _helpers.from_bytes(jwt.encode(self._signer, payload))