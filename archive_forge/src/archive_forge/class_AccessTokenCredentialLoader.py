import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Union
from absl import app
from absl import flags
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
import httplib2
import oauth2client_4_0
import oauth2client_4_0.contrib
import oauth2client_4_0.contrib.gce
import oauth2client_4_0.contrib.multiprocess_file_storage
import oauth2client_4_0.file
import oauth2client_4_0.service_account
import oauth2client_4_0.tools
import requests
import bq_auth_flags
import bq_utils
import wrapped_credentials
from utils import bq_error
class AccessTokenCredentialLoader(CredentialLoader):
    """Credential loader for OAuth access token."""

    def __init__(self, access_token: str, *args, **kwargs) -> None:
        """Creates ApplicationDefaultCredentialFileLoader instance.

    Args:
      access_token: OAuth access token.
      *args: additional arguments to apply to base class.
      **kwargs: additional keyword arguments to apply to base class.
    """
        super(AccessTokenCredentialLoader, self).__init__(*args, **kwargs)
        self._access_token = access_token

    def _Load(self) -> WrappedCredentialsUnionType:
        return oauth2client_4_0.client.AccessTokenCredentials(self._access_token, _CLIENT_USER_AGENT)