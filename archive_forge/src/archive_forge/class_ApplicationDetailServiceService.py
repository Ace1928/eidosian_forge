from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.testing.v1 import testing_v1_messages as messages
class ApplicationDetailServiceService(base_api.BaseApiService):
    """Service class for the applicationDetailService resource."""
    _NAME = 'applicationDetailService'

    def __init__(self, client):
        super(TestingV1.ApplicationDetailServiceService, self).__init__(client)
        self._upload_configs = {}

    def GetApkDetails(self, request, global_params=None):
        """Gets the details of an Android application APK.

      Args:
        request: (TestingApplicationDetailServiceGetApkDetailsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetApkDetailsResponse) The response message.
      """
        config = self.GetMethodConfig('GetApkDetails')
        return self._RunMethod(config, request, global_params=global_params)
    GetApkDetails.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='testing.applicationDetailService.getApkDetails', ordered_params=[], path_params=[], query_params=['bundleLocation_gcsPath'], relative_path='v1/applicationDetailService/getApkDetails', request_field='fileReference', request_type_name='TestingApplicationDetailServiceGetApkDetailsRequest', response_type_name='GetApkDetailsResponse', supports_download=False)