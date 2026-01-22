import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
class ProjectsServiceAccountService(base_api.BaseApiService):
    """Service class for the projects_serviceAccount resource."""
    _NAME = u'projects_serviceAccount'

    def __init__(self, client):
        super(StorageV1.ProjectsServiceAccountService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get the email address of this project's Google Cloud Storage service account.

      Args:
        request: (StorageProjectsServiceAccountGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccount) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.projects.serviceAccount.get', ordered_params=[u'projectId'], path_params=[u'projectId'], query_params=[u'userProject'], relative_path=u'projects/{projectId}/serviceAccount', request_field='', request_type_name=u'StorageProjectsServiceAccountGetRequest', response_type_name=u'ServiceAccount', supports_download=False)