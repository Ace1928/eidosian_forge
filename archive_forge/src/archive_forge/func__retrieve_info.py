import logging
import warnings
from six.moves import http_client
from oauth2client import client
from oauth2client.contrib import _metadata
def _retrieve_info(self, http):
    """Retrieves service account info for invalid credentials.

        Args:
            http: an object to be used to make HTTP requests.
        """
    if self.invalid:
        info = _metadata.get_service_account_info(http, service_account=self.service_account_email or 'default')
        self.invalid = False
        self.service_account_email = info['email']
        self.scopes = info['scopes']