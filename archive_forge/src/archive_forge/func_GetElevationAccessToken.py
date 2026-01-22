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
def GetElevationAccessToken(self, service_account_id, scopes):
    if ',' in service_account_id:
        raise InvalidImpersonationAccount('More than one service accounts were specified, which is not supported. If being set, please unset the auth/disable_load_google_auth property and retry.')
    response = GenerateAccessToken(service_account_id, scopes)
    return ImpersonationCredentials(service_account_id, response.accessToken, response.expireTime, scopes)