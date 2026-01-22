import base64
import datetime
import json
import os
import unittest
import mock
from mock import patch
from six.moves import http_client
from six.moves import urllib
from oauth2client import client
from oauth2client import client
from google_reauth import reauth
from google_reauth import errors
from google_reauth import reauth_creds
from google_reauth import _reauth_client
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
def _get_creds(self):
    return Oauth2WithReauthCredentials(access_token='old_token', client_id='id', client_secret='secret', refresh_token='old_refresh_token', token_expiry=datetime.datetime(2018, 3, 2, 21, 26, 13), token_uri='token_uri', user_agent='user_agent')