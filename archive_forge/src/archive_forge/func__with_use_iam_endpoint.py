import copy
import datetime
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import credentials
from google.auth import jwt
from google.oauth2 import _client
def _with_use_iam_endpoint(self, use_iam_endpoint):
    """Create a copy of these credentials with the use_iam_endpoint value.

        Args:
            use_iam_endpoint (bool): If True, IAM generateIdToken endpoint will
                be used instead of the token_uri. Note that
                iam.serviceAccountTokenCreator role is required to use the IAM
                endpoint. The default value is False. This feature is currently
                experimental and subject to change without notice.

        Returns:
            google.auth.service_account.IDTokenCredentials: A new credentials
                instance.
        """
    cred = self.__class__(self._signer, service_account_email=self._service_account_email, token_uri=self._token_uri, target_audience=self._target_audience, additional_claims=self._additional_claims.copy(), quota_project_id=self.quota_project_id)
    cred._use_iam_endpoint = use_iam_endpoint
    return cred