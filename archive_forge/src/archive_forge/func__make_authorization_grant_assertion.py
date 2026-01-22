import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import iam
from google.auth import jwt
from google.auth.compute_engine import _metadata
from google.oauth2 import _client
def _make_authorization_grant_assertion(self):
    """Create the OAuth 2.0 assertion.
        This assertion is used during the OAuth 2.0 grant to acquire an
        ID token.
        Returns:
            bytes: The authorization grant assertion.
        """
    now = _helpers.utcnow()
    lifetime = datetime.timedelta(seconds=_DEFAULT_TOKEN_LIFETIME_SECS)
    expiry = now + lifetime
    payload = {'iat': _helpers.datetime_to_secs(now), 'exp': _helpers.datetime_to_secs(expiry), 'iss': self.service_account_email, 'aud': self._token_uri, 'target_audience': self._target_audience}
    payload.update(self._additional_claims)
    token = jwt.encode(self._signer, payload)
    return token