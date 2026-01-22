import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import iam
from google.auth import jwt
from google.auth.compute_engine import _metadata
from google.oauth2 import _client
def _call_metadata_identity_endpoint(self, request):
    """Request ID token from metadata identity endpoint.

        Args:
            request (google.auth.transport.Request): The object used to make
                HTTP requests.

        Returns:
            Tuple[str, datetime.datetime]: The ID token and the expiry of the ID token.

        Raises:
            google.auth.exceptions.RefreshError: If the Compute Engine metadata
                service can't be reached or if the instance has no credentials.
            ValueError: If extracting expiry from the obtained ID token fails.
        """
    try:
        path = 'instance/service-accounts/default/identity'
        params = {'audience': self._target_audience, 'format': 'full'}
        id_token = _metadata.get(request, path, params=params)
    except exceptions.TransportError as caught_exc:
        new_exc = exceptions.RefreshError(caught_exc)
        six.raise_from(new_exc, caught_exc)
    _, payload, _, _ = jwt._unverified_decode(id_token)
    return (id_token, datetime.datetime.fromtimestamp(payload['exp']))