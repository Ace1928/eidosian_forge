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
def _LoadFromCache(self) -> Optional['wrapped_credentials.WrappedCredentials']:
    """Loads credential from cache file."""
    if not os.path.exists(self.credential_cache_file):
        return None
    try:
        creds = self._storage.get()
        if not creds:
            legacy_storage = oauth2client_4_0.file.Storage(self.credential_cache_file)
            creds = legacy_storage.get()
            if creds:
                self._storage.put(creds)
    except BaseException as e:
        self._RaiseCredentialsCorrupt(e)
    if not creds:
        return None
    if isinstance(creds, wrapped_credentials.WrappedCredentials):
        scopes = bq_utils.GetClientScopesFor3pi()
    else:
        scopes = bq_utils.GetClientScopesFromFlags()
    if not creds.has_scopes(scopes):
        return None
    return creds