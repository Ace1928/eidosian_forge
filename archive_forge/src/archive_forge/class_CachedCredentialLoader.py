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
class CachedCredentialLoader(CredentialLoader):
    """Base class to add cache capability to credential loader.

  It will attempt to load credential from local cache file first before calling
  derived class to load credential from source. Once credential is retrieved, it
  will save to local cache file for future use.
  """

    def __init__(self, credential_cache_file: str, read_cache_first: bool=True) -> None:
        """Creates CachedCredentialLoader instance.

    Args:
      credential_cache_file: path to a local file to cache credential.
      read_cache_first: whether to load credential from cache first.

    Raises:
      BigqueryError: if cache file cannot be created to store credential.
    """
        logging.info('Loading credentials with the CachedCredentialLoader')
        self.credential_cache_file = credential_cache_file
        self._read_cache_first = read_cache_first
        self._scopes_key = ','.join(sorted(bq_utils.GetClientScopesFromFlags()))
        try:
            self._storage = oauth2client_4_0.contrib.multiprocess_file_storage.MultiprocessFileStorage(credential_cache_file, self._scopes_key)
        except OSError as e:
            raise bq_error.BigqueryError('Cannot create credential file %s: %s' % (credential_cache_file, e))

    @property
    def storage(self) -> 'oauth2client_4_0.contrib.multiprocess_file_storage.MultiprocessFileStorage':
        return self._storage

    def Load(self) -> WrappedCredentialsUnionType:
        cred = self._LoadFromCache() if self._read_cache_first else None
        if cred:
            return cred
        cred = super(CachedCredentialLoader, self).Load()
        if not cred:
            return None
        self._storage.put(cred)
        cred.set_store(self._storage)
        return cred

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

    def _RaiseCredentialsCorrupt(self, e: 'BaseException') -> None:
        bq_utils.ProcessError(e, name='GetCredentialsFromFlags', message_prefix='Credentials appear corrupt. Please delete the credential file and try your command again. You can delete your credential file using "bq init --delete_credentials".\n\nIf that does not work, you may have encountered a bug in the BigQuery CLI.')
        sys.exit(1)