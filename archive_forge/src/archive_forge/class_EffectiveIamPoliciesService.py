from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
class EffectiveIamPoliciesService(base_api.BaseApiService):
    """Service class for the effectiveIamPolicies resource."""
    _NAME = 'effectiveIamPolicies'

    def __init__(self, client):
        super(CloudassetV1.EffectiveIamPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def BatchGet(self, request, global_params=None):
        """Gets effective IAM policies for a batch of resources.

      Args:
        request: (CloudassetEffectiveIamPoliciesBatchGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchGetEffectiveIamPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('BatchGet')
        return self._RunMethod(config, request, global_params=global_params)
    BatchGet.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/effectiveIamPolicies:batchGet', http_method='GET', method_id='cloudasset.effectiveIamPolicies.batchGet', ordered_params=['scope'], path_params=['scope'], query_params=['names'], relative_path='v1/{+scope}/effectiveIamPolicies:batchGet', request_field='', request_type_name='CloudassetEffectiveIamPoliciesBatchGetRequest', response_type_name='BatchGetEffectiveIamPoliciesResponse', supports_download=False)