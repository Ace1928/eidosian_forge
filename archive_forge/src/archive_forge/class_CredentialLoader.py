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
class CredentialLoader(object):
    """Base class for credential loader."""

    def Load(self) -> WrappedCredentialsUnionType:
        """Loads credential."""
        cred = self._Load()
        cred._user_agent = _CLIENT_USER_AGENT
        return cred

    def _Load(self) -> WrappedCredentialsUnionType:
        raise NotImplementedError()