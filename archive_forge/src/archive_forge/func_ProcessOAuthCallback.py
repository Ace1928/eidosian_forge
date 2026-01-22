from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
def ProcessOAuthCallback(self, request, global_params=None):
    """ProcessOAuthCallback fulfills the last leg of the OAuth dance with a source provider. For GitHub this is as defined by https://developer.github.com/apps/building-oauth-apps/authorizing-oauth-apps/#2-users-are-redirected-back-to-your-site-by-github Users will not be able to call this in any meaningful way since they don't have access to the OAuth code used in the exchange. For now, this rpc only supports GitHubEnterprise, but will eventually replace GenerateGitHubAccessToken.

      Args:
        request: (CloudbuildOauthProcessOAuthCallbackRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('ProcessOAuthCallback')
    return self._RunMethod(config, request, global_params=global_params)