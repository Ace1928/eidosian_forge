from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsDicomStoresService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dicomStores resource."""
    _NAME = 'projects_locations_datasets_dicomStores'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsDicomStoresService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new DICOM store within the parent dataset.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DicomStore) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores', http_method='POST', method_id='healthcare.projects.locations.datasets.dicomStores.create', ordered_params=['parent'], path_params=['parent'], query_params=['dicomStoreId'], relative_path='v1alpha2/{+parent}/dicomStores', request_field='dicomStore', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresCreateRequest', response_type_name='DicomStore', supports_download=False)

    def Deidentify(self, request, global_params=None):
        """De-identifies data from the source store and writes it to the destination store. The metadata field type is OperationMetadata. If the request is successful, the response field type is DeidentifyDicomStoreSummary. The LRO result may still be successful if de-identification fails for some DICOM instances. The output DICOM store will not contain these failed resources. The number of resources processed are tracked in Operation.metadata. Error details are logged to Cloud Logging. For more information, see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresDeidentifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Deidentify')
        return self._RunMethod(config, request, global_params=global_params)
    Deidentify.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}:deidentify', http_method='POST', method_id='healthcare.projects.locations.datasets.dicomStores.deidentify', ordered_params=['sourceStore'], path_params=['sourceStore'], query_params=[], relative_path='v1alpha2/{+sourceStore}:deidentify', request_field='deidentifyDicomStoreRequest', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresDeidentifyRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified DICOM store and removes all images that are contained within it.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.dicomStores.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresDeleteRequest', response_type_name='Empty', supports_download=False)

    def Export(self, request, global_params=None):
        """Exports data to the specified destination by copying it from the DICOM store. Errors are also logged to Cloud Logging. For more information, see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging). The metadata field type is OperationMetadata.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}:export', http_method='POST', method_id='healthcare.projects.locations.datasets.dicomStores.export', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:export', request_field='exportDicomDataRequest', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresExportRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified DICOM store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DicomStore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresGetRequest', response_type_name='DicomStore', supports_download=False)

    def GetDICOMStoreMetrics(self, request, global_params=None):
        """Gets metrics associated with the DICOM store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresGetDICOMStoreMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DicomStoreMetrics) The response message.
      """
        config = self.GetMethodConfig('GetDICOMStoreMetrics')
        return self._RunMethod(config, request, global_params=global_params)
    GetDICOMStoreMetrics.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}:getDICOMStoreMetrics', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.getDICOMStoreMetrics', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:getDICOMStoreMetrics', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresGetDICOMStoreMetricsRequest', response_type_name='DicomStoreMetrics', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}:getIamPolicy', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports data into the DICOM store by copying it from the specified source. Errors are logged to Cloud Logging. For more information, see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging). The metadata field type is OperationMetadata.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}:import', http_method='POST', method_id='healthcare.projects.locations.datasets.dicomStores.import', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:import', request_field='importDicomDataRequest', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresImportRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the DICOM stores in the given dataset.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDicomStoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/dicomStores', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresListRequest', response_type_name='ListDicomStoresResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified DICOM store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DicomStore) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}', http_method='PATCH', method_id='healthcare.projects.locations.datasets.dicomStores.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='dicomStore', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresPatchRequest', response_type_name='DicomStore', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}:setIamPolicy', http_method='POST', method_id='healthcare.projects.locations.datasets.dicomStores.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}:testIamPermissions', http_method='POST', method_id='healthcare.projects.locations.datasets.dicomStores.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)