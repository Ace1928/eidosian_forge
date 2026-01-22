from __future__ import absolute_import
import six
from google.auth import _helpers
import google.auth.app_engine
import google.auth.compute_engine
import google.oauth2.credentials
import google.oauth2.service_account
def _convert_gce_app_assertion_credentials(credentials):
    """Converts to :class:`google.auth.compute_engine.Credentials`.

    Args:
        credentials (oauth2client.contrib.gce.AppAssertionCredentials): The
            credentials to convert.

    Returns:
        google.oauth2.service_account.Credentials: The converted credentials.
    """
    return google.auth.compute_engine.Credentials(service_account_email=credentials.service_account_email)