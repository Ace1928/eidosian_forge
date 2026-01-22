from __future__ import absolute_import
from apitools.base.py import base_api
from samples.iam_sample.iam_v1 import iam_v1_messages as messages
class IamPoliciesService(base_api.BaseApiService):
    """Service class for the iamPolicies resource."""
    _NAME = u'iamPolicies'

    def __init__(self, client):
        super(IamV1.IamPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def GetPolicyDetails(self, request, global_params=None):
        """Returns the current IAM policy and the policies on the inherited resources.
that the user has access to.

      Args:
        request: (GetPolicyDetailsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetPolicyDetailsResponse) The response message.
      """
        config = self.GetMethodConfig('GetPolicyDetails')
        return self._RunMethod(config, request, global_params=global_params)
    GetPolicyDetails.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'iam.iamPolicies.getPolicyDetails', ordered_params=[], path_params=[], query_params=[], relative_path=u'v1/iamPolicies:getPolicyDetails', request_field='<request>', request_type_name=u'GetPolicyDetailsRequest', response_type_name=u'GetPolicyDetailsResponse', supports_download=False)