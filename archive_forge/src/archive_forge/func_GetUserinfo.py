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
def GetUserinfo(credentials, http=None):
    """Get the userinfo associated with the given credentials.

    This is dependent on the token having either the userinfo.email or
    userinfo.profile scope for the given token.

    Args:
      credentials: (oauth2client.client.Credentials) incoming credentials
      http: (httplib2.Http, optional) http instance to use

    Returns:
      The email address for this token, or None if the required scopes
      aren't available.
    """
    http = http or httplib2.Http()
    url = _GetUserinfoUrl(credentials)
    response, content = http.request(url)
    if response.status == http_client.BAD_REQUEST:
        credentials.refresh(http)
        url = _GetUserinfoUrl(credentials)
        response, content = http.request(url)
    return json.loads(content or '{}')