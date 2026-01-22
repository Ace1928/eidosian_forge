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
def ServiceAccountCredentialsFromP12File(service_account_name, private_key_filename, scopes, user_agent):
    """Create a new credential from the named .p12 keyfile."""
    private_key_filename = os.path.expanduser(private_key_filename)
    scopes = util.NormalizeScopes(scopes)
    if oauth2client.__version__ > '1.5.2':
        credentials = service_account.ServiceAccountCredentials.from_p12_keyfile(service_account_name, private_key_filename, scopes=scopes)
        if credentials is not None:
            credentials.user_agent = user_agent
        return credentials
    else:
        with open(private_key_filename, 'rb') as key_file:
            return oauth2client.client.SignedJwtAssertionCredentials(service_account_name, key_file.read(), scopes, user_agent=user_agent)