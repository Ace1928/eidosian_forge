from __future__ import absolute_import
import io
import logging
import os
import random
import re
import time
import urllib
import httplib2
from oauth2client import client
from oauth2client import file as oauth2client_file
from oauth2client import tools
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.tools.value_mixin import ValueMixin
from googlecloudsdk.third_party.appengine._internal import six_subset
class OAuth2Parameters(ValueMixin):
    """Class encapsulating parameters related to OAuth2 authentication."""

    def __init__(self, access_token, client_id, client_secret, scope, refresh_token, credential_file, token_uri=None, credentials=None):
        self.access_token = access_token
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.refresh_token = refresh_token
        self.credential_file = credential_file
        self.token_uri = token_uri
        self.credentials = credentials