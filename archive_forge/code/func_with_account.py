from datetime import datetime
import io
import json
import logging
import six
from google.auth import _cloud_sdk
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import reauth
def with_account(self, account):
    """Create a new instance with the given account.

        Args:
            account (str): Account to get the access token for.

        Returns:
            google.oauth2.credentials.UserAccessTokenCredentials: The created
                credentials with the given account.
        """
    return self.__class__(account=account, quota_project_id=self._quota_project_id)