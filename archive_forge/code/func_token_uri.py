from datetime import datetime
import io
import json
import logging
import six
from google.auth import _cloud_sdk
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import reauth
@property
def token_uri(self):
    """Optional[str]: The OAuth 2.0 authorization server's token endpoint
        URI."""
    return self._token_uri