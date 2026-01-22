from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import transport
from oauth2client import client
def GetElevationAccessTokenGoogleAuth(self, source_credentials, target_principal, delegates, scopes):
    """Creates a fresh impersonation credential using google-auth library."""
    from google.auth import exceptions as google_auth_exceptions
    from google.auth import impersonated_credentials as google_auth_impersonated_credentials
    from googlecloudsdk.core import requests as core_requests
    request_client = core_requests.GoogleAuthRequest()
    source_credentials.refresh(request_client)
    cred = google_auth_impersonated_credentials.Credentials(source_credentials=source_credentials, target_principal=target_principal, target_scopes=scopes, delegates=delegates)
    self.PerformIamEndpointsOverride()
    try:
        cred.refresh(request_client)
    except google_auth_exceptions.RefreshError:
        raise ImpersonatedCredGoogleAuthRefreshError('Failed to impersonate [{service_acc}]. Make sure the account that\'s trying to impersonate it has access to the service account itself and the "roles/iam.serviceAccountTokenCreator" role.'.format(service_acc=target_principal))
    return cred