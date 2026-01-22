from __future__ import print_function
import argparse
import contextlib
import datetime
import json
import os
import threading
import warnings
import httplib2
import oauth2client
import oauth2client.client
from oauth2client import service_account
from oauth2client import tools  # for gflags declarations
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.py import exceptions
from apitools.base.py import util
def _WriteCacheFile(self, cache_filename, scopes):
    """Writes the credential metadata to the cache file.

        This does not save the credentials themselves (CredentialStore class
        optionally handles that after this class is initialized).

        Args:
          cache_filename: Cache filename to check.
          scopes: Scopes for the desired credentials.
        """
    scopes = sorted([six.ensure_text(scope) for scope in scopes])
    creds = {'scopes': scopes, 'svc_acct_name': self.__service_account_name}
    creds_str = json.dumps(creds)
    cache_file = _MultiProcessCacheFile(cache_filename)
    try:
        cache_file.LockedWrite(creds_str)
    except KeyboardInterrupt:
        raise
    except:
        pass