from __future__ import print_function
import httplib2
import json
import os
from select import select
import stat
from sys import stdin
import time
import webbrowser
from base64 import (
from six.moves.urllib.parse import parse_qs
from lazr.restfulclient.errors import HTTPError
from lazr.restfulclient.authorize.oauth import (
from launchpadlib import uris
def wait_for_end_user_authorization(self, credentials):
    """Wait for the end-user to authorize"""
    self.output(self.WAITING_FOR_LAUNCHPAD)
    start_time = time.time()
    while credentials.access_token is None:
        time.sleep(access_token_poll_time)
        try:
            if self.check_end_user_authorization(credentials):
                break
        except EndUserNoAuthorization:
            pass
        if time.time() >= start_time + access_token_poll_timeout:
            raise TokenAuthorizationTimedOut('Timed out after %d seconds.' % access_token_poll_timeout)