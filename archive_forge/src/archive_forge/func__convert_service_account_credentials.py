from __future__ import absolute_import
import six
from google.auth import _helpers
import google.auth.app_engine
import google.auth.compute_engine
import google.oauth2.credentials
import google.oauth2.service_account
def _convert_service_account_credentials(credentials):
    """Converts to :class:`google.oauth2.service_account.Credentials`.

    Args:
        credentials (Union[
            oauth2client.service_account.ServiceAccountCredentials,
            oauth2client.service_account._JWTAccessCredentials]): The
            credentials to convert.

    Returns:
        google.oauth2.service_account.Credentials: The converted credentials.
    """
    info = credentials.serialization_data.copy()
    info['token_uri'] = credentials.token_uri
    return google.oauth2.service_account.Credentials.from_service_account_info(info)