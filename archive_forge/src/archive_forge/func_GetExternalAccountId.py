from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from google.oauth2 import utils as oauth2_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from six.moves import http_client
from six.moves import urllib
def GetExternalAccountId(creds):
    """Returns the external account credentials' identifier.

  This requires basic client authentication and only works with external
  account credentials that have not been impersonated. The returned username
  field is used for the account ID.

  Args:
    creds (google.auth.external_account.Credentials): The external account
      credentials whose account ID is to be determined.

  Returns:
    Optional(str): The account ID string if determinable.

  Raises:
    InactiveCredentialsError: If the credentials are invalid or expired.
    TokenIntrospectionError: If an error is encountered while calling the
      token introspection endpoint.
  """
    from googlecloudsdk.core import requests as core_requests
    client_authentication = oauth2_utils.ClientAuthentication(oauth2_utils.ClientAuthType.basic, config.CLOUDSDK_CLIENT_ID, config.CLOUDSDK_CLIENT_NOTSOSECRET)
    token_introspection_endpoint = _EXTERNAL_ACCT_TOKEN_INTROSPECT_ENDPOINT
    endpoint_override = properties.VALUES.auth.token_introspection_endpoint.Get()
    property_override = creds.token_info_url
    if endpoint_override or property_override:
        token_introspection_endpoint = endpoint_override or property_override
    oauth_introspection = IntrospectionClient(token_introspect_endpoint=token_introspection_endpoint, client_authentication=client_authentication)
    request = core_requests.GoogleAuthRequest()
    if not creds.valid:
        creds.refresh(request)
    token_info = oauth_introspection.introspect(request, creds.token)
    return token_info.get('username')