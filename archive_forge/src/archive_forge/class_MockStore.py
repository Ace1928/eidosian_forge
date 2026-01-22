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
class MockStore(client.Storage):

    def __init__(self):
        self.credentials = None

    def locked_put(self, credentials):
        self.credentials = credentials

    def locked_get(self):
        return self.credentials