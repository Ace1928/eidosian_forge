import datetime
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import credentials
from google.auth import exceptions
from google.auth import jwt
from google.oauth2 import _client
def with_gdch_audience(self, audience):
    """Create a copy of GDCH credentials with the specified audience.

        Args:
            audience (str): The intended audience for GDCH credentials.
        """
    return self.__class__(self._signer, self._service_identity_name, self._project, audience, self._token_uri, self._ca_cert_path)