from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
def Receive(self, request, global_params=None):
    """ReceiveGitHubDotComWebhook is called when the API receives a github.com webhook.

      Args:
        request: (CloudbuildGithubDotComWebhookReceiveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('Receive')
    return self._RunMethod(config, request, global_params=global_params)