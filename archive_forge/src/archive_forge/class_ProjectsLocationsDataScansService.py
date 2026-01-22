from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
class ProjectsLocationsDataScansService(base_api.BaseApiService):
    """Service class for the projects_locations_dataScans resource."""
    _NAME = 'projects_locations_dataScans'

    def __init__(self, client):
        super(DataplexV1.ProjectsLocationsDataScansService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a DataScan resource.

      Args:
        request: (DataplexProjectsLocationsDataScansCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans', http_method='POST', method_id='dataplex.projects.locations.dataScans.create', ordered_params=['parent'], path_params=['parent'], query_params=['dataScanId', 'validateOnly'], relative_path='v1/{+parent}/dataScans', request_field='googleCloudDataplexV1DataScan', request_type_name='DataplexProjectsLocationsDataScansCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a DataScan resource.

      Args:
        request: (DataplexProjectsLocationsDataScansDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}', http_method='DELETE', method_id='dataplex.projects.locations.dataScans.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsDataScansDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def GenerateDataQualityRules(self, request, global_params=None):
        """Generates recommended DataQualityRule from a data profiling DataScan.

      Args:
        request: (DataplexProjectsLocationsDataScansGenerateDataQualityRulesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1GenerateDataQualityRulesResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateDataQualityRules')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateDataQualityRules.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}:generateDataQualityRules', http_method='POST', method_id='dataplex.projects.locations.dataScans.generateDataQualityRules', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:generateDataQualityRules', request_field='googleCloudDataplexV1GenerateDataQualityRulesRequest', request_type_name='DataplexProjectsLocationsDataScansGenerateDataQualityRulesRequest', response_type_name='GoogleCloudDataplexV1GenerateDataQualityRulesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a DataScan resource.

      Args:
        request: (DataplexProjectsLocationsDataScansGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1DataScan) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}', http_method='GET', method_id='dataplex.projects.locations.dataScans.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsDataScansGetRequest', response_type_name='GoogleCloudDataplexV1DataScan', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DataplexProjectsLocationsDataScansGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}:getIamPolicy', http_method='GET', method_id='dataplex.projects.locations.dataScans.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='DataplexProjectsLocationsDataScansGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DataScans.

      Args:
        request: (DataplexProjectsLocationsDataScansListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1ListDataScansResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans', http_method='GET', method_id='dataplex.projects.locations.dataScans.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/dataScans', request_field='', request_type_name='DataplexProjectsLocationsDataScansListRequest', response_type_name='GoogleCloudDataplexV1ListDataScansResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a DataScan resource.

      Args:
        request: (DataplexProjectsLocationsDataScansPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}', http_method='PATCH', method_id='dataplex.projects.locations.dataScans.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='googleCloudDataplexV1DataScan', request_type_name='DataplexProjectsLocationsDataScansPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Run(self, request, global_params=None):
        """Runs an on-demand execution of a DataScan.

      Args:
        request: (DataplexProjectsLocationsDataScansRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1RunDataScanResponse) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}:run', http_method='POST', method_id='dataplex.projects.locations.dataScans.run', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:run', request_field='googleCloudDataplexV1RunDataScanRequest', request_type_name='DataplexProjectsLocationsDataScansRunRequest', response_type_name='GoogleCloudDataplexV1RunDataScanResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.Can return NOT_FOUND, INVALID_ARGUMENT, and PERMISSION_DENIED errors.

      Args:
        request: (DataplexProjectsLocationsDataScansSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}:setIamPolicy', http_method='POST', method_id='dataplex.projects.locations.dataScans.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='DataplexProjectsLocationsDataScansSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DataplexProjectsLocationsDataScansTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}:testIamPermissions', http_method='POST', method_id='dataplex.projects.locations.dataScans.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='DataplexProjectsLocationsDataScansTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)