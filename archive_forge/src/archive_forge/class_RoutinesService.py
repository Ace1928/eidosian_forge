from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigquery.v2 import bigquery_v2_messages as messages
class RoutinesService(base_api.BaseApiService):
    """Service class for the routines resource."""
    _NAME = 'routines'

    def __init__(self, client):
        super(BigqueryV2.RoutinesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the routine specified by routineId from the dataset.

      Args:
        request: (BigqueryRoutinesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BigqueryRoutinesDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines/{routinesId}', http_method='DELETE', method_id='bigquery.routines.delete', ordered_params=['projectId', 'datasetId', 'routineId'], path_params=['datasetId', 'projectId', 'routineId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/routines/{+routineId}', request_field='', request_type_name='BigqueryRoutinesDeleteRequest', response_type_name='BigqueryRoutinesDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified routine resource by routine ID.

      Args:
        request: (BigqueryRoutinesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Routine) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines/{routinesId}', http_method='GET', method_id='bigquery.routines.get', ordered_params=['projectId', 'datasetId', 'routineId'], path_params=['datasetId', 'projectId', 'routineId'], query_params=['readMask'], relative_path='projects/{+projectId}/datasets/{+datasetId}/routines/{+routineId}', request_field='', request_type_name='BigqueryRoutinesGetRequest', response_type_name='Routine', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (BigqueryRoutinesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines/{routinesId}:getIamPolicy', http_method='POST', method_id='bigquery.routines.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='BigqueryRoutinesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new routine in the dataset.

      Args:
        request: (BigqueryRoutinesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Routine) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines', http_method='POST', method_id='bigquery.routines.insert', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/routines', request_field='routine', request_type_name='BigqueryRoutinesInsertRequest', response_type_name='Routine', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all routines in the specified dataset. Requires the READER dataset role.

      Args:
        request: (BigqueryRoutinesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRoutinesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines', http_method='GET', method_id='bigquery.routines.list', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=['filter', 'maxResults', 'pageToken', 'readMask'], relative_path='projects/{+projectId}/datasets/{+datasetId}/routines', request_field='', request_type_name='BigqueryRoutinesListRequest', response_type_name='ListRoutinesResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (BigqueryRoutinesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines/{routinesId}:setIamPolicy', http_method='POST', method_id='bigquery.routines.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='BigqueryRoutinesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates information in an existing routine. The update method replaces the entire Routine resource.

      Args:
        request: (BigqueryRoutinesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Routine) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/routines/{routinesId}', http_method='PUT', method_id='bigquery.routines.update', ordered_params=['projectId', 'datasetId', 'routineId'], path_params=['datasetId', 'projectId', 'routineId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/routines/{+routineId}', request_field='routine', request_type_name='BigqueryRoutinesUpdateRequest', response_type_name='Routine', supports_download=False)