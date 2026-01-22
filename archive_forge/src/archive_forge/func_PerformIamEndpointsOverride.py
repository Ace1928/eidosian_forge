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
@classmethod
def PerformIamEndpointsOverride(cls):
    """Perform IAM endpoint override if needed.

    We will override IAM generateAccessToken, signBlob, and generateIdToken
    endpoint under the following conditions.
    (1) If the [api_endpoint_overrides/iamcredentials] property is explicitly
    set, we replace "https://iamcredentials.googleapis.com/" with the given
    property value in these endpoints.
    (2) If the property above is not set, and the [core/universe_domain] value
    is not default, we replace "googleapis.com" with the [core/universe_domain]
    property value in these endpoints.
    """
    from google.auth import impersonated_credentials as google_auth_impersonated_credentials
    effective_iam_endpoint = GetEffectiveIamEndpoint()
    google_auth_impersonated_credentials._IAM_ENDPOINT = google_auth_impersonated_credentials._IAM_ENDPOINT.replace(IAM_ENDPOINT_GDU, effective_iam_endpoint)
    google_auth_impersonated_credentials._IAM_SIGN_ENDPOINT = google_auth_impersonated_credentials._IAM_SIGN_ENDPOINT.replace(IAM_ENDPOINT_GDU, effective_iam_endpoint)
    google_auth_impersonated_credentials._IAM_IDTOKEN_ENDPOINT = google_auth_impersonated_credentials._IAM_IDTOKEN_ENDPOINT.replace(IAM_ENDPOINT_GDU, effective_iam_endpoint)