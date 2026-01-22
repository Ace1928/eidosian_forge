from the current environment without the need to copy, save and manage
import abc
import copy
import datetime
import io
import json
import re
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import metrics
from google.oauth2 import sts
from google.oauth2 import utils
def _initialize_impersonated_credentials(self):
    """Generates an impersonated credentials.

        For more details, see `projects.serviceAccounts.generateAccessToken`_.

        .. _projects.serviceAccounts.generateAccessToken: https://cloud.google.com/iam/docs/reference/credentials/rest/v1/projects.serviceAccounts/generateAccessToken

        Returns:
            impersonated_credentials.Credential: The impersonated credentials
                object.

        Raises:
            google.auth.exceptions.RefreshError: If the generateAccessToken
                endpoint returned an error.
        """
    kwargs = self._constructor_args()
    kwargs.update(service_account_impersonation_url=None, service_account_impersonation_options={})
    source_credentials = self.__class__(**kwargs)
    source_credentials._metrics_options = self._metrics_options
    target_principal = self.service_account_email
    if not target_principal:
        raise exceptions.RefreshError('Unable to determine target principal from service account impersonation URL.')
    scopes = self._scopes if self._scopes is not None else self._default_scopes
    return impersonated_credentials.Credentials(source_credentials=source_credentials, target_principal=target_principal, target_scopes=scopes, quota_project_id=self._quota_project_id, iam_endpoint_override=self._service_account_impersonation_url, lifetime=self._service_account_impersonation_options.get('token_lifetime_seconds'))