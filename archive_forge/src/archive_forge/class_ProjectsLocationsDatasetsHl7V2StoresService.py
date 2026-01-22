from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsHl7V2StoresService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_hl7V2Stores resource."""
    _NAME = 'projects_locations_datasets_hl7V2Stores'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsHl7V2StoresService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new HL7v2 store within the parent dataset.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Hl7V2Store) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores', http_method='POST', method_id='healthcare.projects.locations.datasets.hl7V2Stores.create', ordered_params=['parent'], path_params=['parent'], query_params=['hl7V2StoreId'], relative_path='v1alpha2/{+parent}/hl7V2Stores', request_field='hl7V2Store', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresCreateRequest', response_type_name='Hl7V2Store', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified HL7v2 store and removes all messages that it contains.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores/{hl7V2StoresId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.hl7V2Stores.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresDeleteRequest', response_type_name='Empty', supports_download=False)

    def Export(self, request, global_params=None):
        """Exports the messages to a destination in the store with transformations. The start and the end time relative to message generation time (MSH.7) can be specified to filter messages in a range instead of exporting all at once. This API returns an Operation that can be used to track the status of the job by calling GetOperation. Immediate fatal errors appear in the error field. Otherwise, when the operation finishes, a detailed response of type ExportMessagesResponse is returned in the response field. The metadata field type for this operation is OperationMetadata.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores/{hl7V2StoresId}:export', http_method='POST', method_id='healthcare.projects.locations.datasets.hl7V2Stores.export', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:export', request_field='exportMessagesRequest', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresExportRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified HL7v2 store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Hl7V2Store) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores/{hl7V2StoresId}', http_method='GET', method_id='healthcare.projects.locations.datasets.hl7V2Stores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresGetRequest', response_type_name='Hl7V2Store', supports_download=False)

    def GetHL7v2StoreMetrics(self, request, global_params=None):
        """Gets metrics associated with the HL7v2 store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresGetHL7v2StoreMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Hl7V2StoreMetrics) The response message.
      """
        config = self.GetMethodConfig('GetHL7v2StoreMetrics')
        return self._RunMethod(config, request, global_params=global_params)
    GetHL7v2StoreMetrics.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores/{hl7V2StoresId}:getHL7v2StoreMetrics', http_method='GET', method_id='healthcare.projects.locations.datasets.hl7V2Stores.getHL7v2StoreMetrics', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:getHL7v2StoreMetrics', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresGetHL7v2StoreMetricsRequest', response_type_name='Hl7V2StoreMetrics', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores/{hl7V2StoresId}:getIamPolicy', http_method='GET', method_id='healthcare.projects.locations.datasets.hl7V2Stores.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Import(self, request, global_params=None):
        """Import messages to the HL7v2 store by loading data from the specified sources. This method is optimized to load large quantities of data using import semantics that ignore some HL7v2 store configuration options and are not suitable for all use cases. It is primarily intended to load data into an empty HL7v2 store that is not being used by other clients. An existing message will be overwritten if a duplicate message is imported. A duplicate message is a message with the same raw bytes as a message that already exists in this HL7v2 store. When a message is overwritten, its labels will also be overwritten. The import operation is idempotent unless the input data contains multiple valid messages with the same raw bytes but different labels. In that case, after the import completes, the store contains exactly one message with those raw bytes but there is no ordering guarantee on which version of the labels it has. The operation result counters do not count duplicated raw bytes as an error and count one success for each message in the input, which might result in a success count larger than the number of messages in the HL7v2 store. If some messages fail to import, for example due to parsing errors, successfully imported messages are not rolled back. This method returns an Operation that can be used to track the status of the import by calling GetOperation. Immediate fatal errors appear in the error field, errors are also logged to Cloud Logging (see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging)). Otherwise, when the operation finishes, a response of type ImportMessagesResponse is returned in the response field. The metadata field type for this operation is OperationMetadata.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores/{hl7V2StoresId}:import', http_method='POST', method_id='healthcare.projects.locations.datasets.hl7V2Stores.import', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:import', request_field='importMessagesRequest', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresImportRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the HL7v2 stores in the given dataset.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListHl7V2StoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores', http_method='GET', method_id='healthcare.projects.locations.datasets.hl7V2Stores.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/hl7V2Stores', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresListRequest', response_type_name='ListHl7V2StoresResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the HL7v2 store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Hl7V2Store) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores/{hl7V2StoresId}', http_method='PATCH', method_id='healthcare.projects.locations.datasets.hl7V2Stores.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='hl7V2Store', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresPatchRequest', response_type_name='Hl7V2Store', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores/{hl7V2StoresId}:setIamPolicy', http_method='POST', method_id='healthcare.projects.locations.datasets.hl7V2Stores.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores/{hl7V2StoresId}:testIamPermissions', http_method='POST', method_id='healthcare.projects.locations.datasets.hl7V2Stores.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)