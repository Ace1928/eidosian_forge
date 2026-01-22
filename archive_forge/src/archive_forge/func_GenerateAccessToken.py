from __future__ import absolute_import
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.iamcredentials_apitools.iamcredentials_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
def GenerateAccessToken(self, request, global_params=None):
    """Generates an OAuth 2.0 access token for a service account.

      Args:
        request: (IamcredentialsProjectsServiceAccountsGenerateAccessTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateAccessTokenResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateAccessToken')
    return self._RunMethod(config, request, global_params=global_params)