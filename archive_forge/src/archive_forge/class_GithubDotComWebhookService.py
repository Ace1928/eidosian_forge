from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class GithubDotComWebhookService(base_api.BaseApiService):
    """Service class for the githubDotComWebhook resource."""
    _NAME = 'githubDotComWebhook'

    def __init__(self, client):
        super(CloudbuildV1.GithubDotComWebhookService, self).__init__(client)
        self._upload_configs = {}

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
    Receive.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.githubDotComWebhook.receive', ordered_params=[], path_params=[], query_params=['webhookKey'], relative_path='v1/githubDotComWebhook:receive', request_field='httpBody', request_type_name='CloudbuildGithubDotComWebhookReceiveRequest', response_type_name='Empty', supports_download=False)