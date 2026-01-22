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
class ServiceAccountPrivateKeyFileLoader(ServiceAccountPrivateKeyLoader):
    """Credential loader for private key stored in a file."""

    def __init__(self, service_account: str, file_path: str, password: str, *args, **kwargs) -> None:
        """Creates ServiceAccountPrivateKeyFileLoader instance.

    Args:
      service_account: service account the private key is for.
      file_path: path to the file containing private key (in P12 format).
      password: password to uncrypt the private key file.
      *args: additional arguments to apply to base class.
      **kwargs: additional keyword arguments to apply to base class.
    """
        super(ServiceAccountPrivateKeyFileLoader, self).__init__(*args, **kwargs)
        self._service_account = service_account
        self._file_path = file_path
        self._password = password

    def _Load(self) -> WrappedCredentialsUnionType:
        try:
            return oauth2client_4_0.service_account.ServiceAccountCredentials.from_p12_keyfile(service_account_email=self._service_account, filename=self._file_path, scopes=bq_utils.GetClientScopesFromFlags(), private_key_password=self._password, token_uri=oauth2client_4_0.GOOGLE_TOKEN_URI, revoke_uri=oauth2client_4_0.GOOGLE_REVOKE_URI)
        except IOError as e:
            raise app.UsageError('Service account specified, but private key in file "%s" cannot be read:\n%s' % (self._file_path, e))