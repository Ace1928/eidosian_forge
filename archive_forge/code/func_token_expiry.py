import copy
import datetime
import io
import json
from google.auth import aws
from google.auth import credentials
from google.auth import exceptions
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import pluggable
from google.auth.transport import requests
from gslib.utils import constants
import oauth2client
@token_expiry.setter
def token_expiry(self, value):
    self._base.expiry = value