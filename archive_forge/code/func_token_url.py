import datetime
import io
import json
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
from google.oauth2 import utils
@property
def token_url(self):
    """Optional[str]: The STS token exchange endpoint for refresh."""
    return self._token_url