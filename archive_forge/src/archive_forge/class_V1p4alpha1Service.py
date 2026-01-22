from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1p4alpha1 import cloudasset_v1p4alpha1_messages as messages
class V1p4alpha1Service(base_api.BaseApiService):
    """Service class for the v1p4alpha1 resource."""
    _NAME = 'v1p4alpha1'

    def __init__(self, client):
        super(CloudassetV1p4alpha1.V1p4alpha1Service, self).__init__(client)
        self._upload_configs = {}

    def AnalyzeIamPolicy(self, request, global_params=None):
        """Analyzes IAM policies based on the specified request. Returns.
a list of IamPolicyAnalysisResult matching the request.

      Args:
        request: (CloudassetAnalyzeIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeIamPolicyResponse) The response message.
      """
        config = self.GetMethodConfig('AnalyzeIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    AnalyzeIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1p4alpha1/{v1p4alpha1Id}/{v1p4alpha1Id1}:analyzeIamPolicy', http_method='GET', method_id='cloudasset.analyzeIamPolicy', ordered_params=['parent'], path_params=['parent'], query_params=['accessSelector_permissions', 'accessSelector_roles', 'identitySelector_identity', 'options_expandGroups', 'options_expandResources', 'options_expandRoles', 'options_maxFanoutsPerGroup', 'options_maxFanoutsPerResource', 'options_outputGroupEdges', 'options_outputPartialResultBeforeTimeout', 'options_outputResourceEdges', 'resourceSelector_fullResourceName'], relative_path='v1p4alpha1/{+parent}:analyzeIamPolicy', request_field='', request_type_name='CloudassetAnalyzeIamPolicyRequest', response_type_name='AnalyzeIamPolicyResponse', supports_download=False)