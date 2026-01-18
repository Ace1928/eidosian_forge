import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import iam
from google.auth import jwt
from google.auth.compute_engine import _metadata
from google.oauth2 import _client
def with_target_audience(self, target_audience):
    """Create a copy of these credentials with the specified target
        audience.
        Args:
            target_audience (str): The intended audience for these credentials,
            used when requesting the ID Token.
        Returns:
            google.auth.service_account.IDTokenCredentials: A new credentials
                instance.
        """
    if self._use_metadata_identity_endpoint:
        return self.__class__(None, target_audience=target_audience, use_metadata_identity_endpoint=True, quota_project_id=self._quota_project_id)
    else:
        return self.__class__(None, service_account_email=self._service_account_email, token_uri=self._token_uri, target_audience=target_audience, additional_claims=self._additional_claims.copy(), signer=self.signer, use_metadata_identity_endpoint=False, quota_project_id=self._quota_project_id)