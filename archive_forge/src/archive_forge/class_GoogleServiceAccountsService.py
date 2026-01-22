from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storagetransfer.v1 import storagetransfer_v1_messages as messages
class GoogleServiceAccountsService(base_api.BaseApiService):
    """Service class for the googleServiceAccounts resource."""
    _NAME = 'googleServiceAccounts'

    def __init__(self, client):
        super(StoragetransferV1.GoogleServiceAccountsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns the Google service account that is used by Storage Transfer Service to access buckets in the project where transfers run or in other projects. Each Google service account is associated with one Google Cloud project. Users should add this service account to the Google Cloud Storage bucket ACLs to grant access to Storage Transfer Service. This service account is created and owned by Storage Transfer Service and can only be used by Storage Transfer Service.

      Args:
        request: (StoragetransferGoogleServiceAccountsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleServiceAccount) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='storagetransfer.googleServiceAccounts.get', ordered_params=['projectId'], path_params=['projectId'], query_params=[], relative_path='v1/googleServiceAccounts/{projectId}', request_field='', request_type_name='StoragetransferGoogleServiceAccountsGetRequest', response_type_name='GoogleServiceAccount', supports_download=False)