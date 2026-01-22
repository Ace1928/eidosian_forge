from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha import osconfig_v1alpha_messages as messages
class ProjectsLocationsOsPolicyAssignmentsOperationsService(base_api.BaseApiService):
    """Service class for the projects_locations_osPolicyAssignments_operations resource."""
    _NAME = 'projects_locations_osPolicyAssignments_operations'

    def __init__(self, client):
        super(OsconfigV1alpha.ProjectsLocationsOsPolicyAssignmentsOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.

      Args:
        request: (OsconfigProjectsLocationsOsPolicyAssignmentsOperationsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/osPolicyAssignments/{osPolicyAssignmentsId}/operations/{operationsId}:cancel', http_method='POST', method_id='osconfig.projects.locations.osPolicyAssignments.operations.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:cancel', request_field='cancelOperationRequest', request_type_name='OsconfigProjectsLocationsOsPolicyAssignmentsOperationsCancelRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (OsconfigProjectsLocationsOsPolicyAssignmentsOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/osPolicyAssignments/{osPolicyAssignmentsId}/operations/{operationsId}', http_method='GET', method_id='osconfig.projects.locations.osPolicyAssignments.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='OsconfigProjectsLocationsOsPolicyAssignmentsOperationsGetRequest', response_type_name='Operation', supports_download=False)