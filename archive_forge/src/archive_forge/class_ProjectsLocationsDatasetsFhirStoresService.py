from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsFhirStoresService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_fhirStores resource."""
    _NAME = 'projects_locations_datasets_fhirStores'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsFhirStoresService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new FHIR store within the parent dataset.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FhirStore) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores', http_method='POST', method_id='healthcare.projects.locations.datasets.fhirStores.create', ordered_params=['parent'], path_params=['parent'], query_params=['fhirStoreId'], relative_path='v1alpha2/{+parent}/fhirStores', request_field='fhirStore', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresCreateRequest', response_type_name='FhirStore', supports_download=False)

    def Deidentify(self, request, global_params=None):
        """De-identifies data from the source store and writes it to the destination store. The metadata field type is OperationMetadata. If the request is successful, the response field type is DeidentifyFhirStoreSummary. The number of resources processed are tracked in Operation.metadata. Error details are logged to Cloud Logging. For more information, see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging).

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresDeidentifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Deidentify')
        return self._RunMethod(config, request, global_params=global_params)
    Deidentify.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores/{fhirStoresId}:deidentify', http_method='POST', method_id='healthcare.projects.locations.datasets.fhirStores.deidentify', ordered_params=['sourceStore'], path_params=['sourceStore'], query_params=[], relative_path='v1alpha2/{+sourceStore}:deidentify', request_field='deidentifyFhirStoreRequest', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresDeidentifyRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified FHIR store and removes all resources within it.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores/{fhirStoresId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.fhirStores.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresDeleteRequest', response_type_name='Empty', supports_download=False)

    def Export(self, request, global_params=None):
        """Export resources from the FHIR store to the specified destination. This method returns an Operation that can be used to track the status of the export by calling GetOperation. Immediate fatal errors appear in the error field, errors are also logged to Cloud Logging (see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging)). Otherwise, when the operation finishes, a detailed response of type ExportResourcesResponse is returned in the response field. The metadata field type for this operation is OperationMetadata.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores/{fhirStoresId}:export', http_method='POST', method_id='healthcare.projects.locations.datasets.fhirStores.export', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:export', request_field='exportResourcesRequest', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresExportRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the configuration of the specified FHIR store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FhirStore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores/{fhirStoresId}', http_method='GET', method_id='healthcare.projects.locations.datasets.fhirStores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresGetRequest', response_type_name='FhirStore', supports_download=False)

    def GetFHIRStoreMetrics(self, request, global_params=None):
        """Gets metrics associated with the FHIR store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresGetFHIRStoreMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FhirStoreMetrics) The response message.
      """
        config = self.GetMethodConfig('GetFHIRStoreMetrics')
        return self._RunMethod(config, request, global_params=global_params)
    GetFHIRStoreMetrics.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores/{fhirStoresId}:getFHIRStoreMetrics', http_method='GET', method_id='healthcare.projects.locations.datasets.fhirStores.getFHIRStoreMetrics', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:getFHIRStoreMetrics', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresGetFHIRStoreMetricsRequest', response_type_name='FhirStoreMetrics', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores/{fhirStoresId}:getIamPolicy', http_method='GET', method_id='healthcare.projects.locations.datasets.fhirStores.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Import(self, request, global_params=None):
        """Import resources to the FHIR store by loading data from the specified sources. This method is optimized to load large quantities of data using import semantics that ignore some FHIR store configuration options and are not suitable for all use cases. It is primarily intended to load data into an empty FHIR store that is not being used by other clients. In cases where this method is not appropriate, consider using ExecuteBundle to load data. Every resource in the input must contain a client-supplied ID. Each resource is stored using the supplied ID regardless of the enable_update_create setting on the FHIR store. It is strongly advised not to include or encode any sensitive data such as patient identifiers in client-specified resource IDs. Those IDs are part of the FHIR resource path recorded in Cloud Audit Logs and Cloud Pub/Sub notifications. Those IDs can also be contained in reference fields within other resources. The import process does not enforce referential integrity, regardless of the disable_referential_integrity setting on the FHIR store. This allows the import of resources with arbitrary interdependencies without considering grouping or ordering, but if the input data contains invalid references or if some resources fail to be imported, the FHIR store might be left in a state that violates referential integrity. The import process does not trigger Pub/Sub notification or BigQuery streaming update, regardless of how those are configured on the FHIR store. If a resource with the specified ID already exists, the most recent version of the resource is overwritten without creating a new historical version, regardless of the disable_resource_versioning setting on the FHIR store. If transient failures occur during the import, successfully imported resources could be overwritten more than once. The import operation is idempotent unless the input data contains multiple valid resources with the same ID but different contents. In that case, after the import completes, the store contains exactly one resource with that ID but there is no ordering guarantee on which version of the contents it has. The operation result counters do not count duplicate IDs as an error and count one success for each resource in the input, which might result in a success count larger than the number of resources in the FHIR store. This often occurs when importing data organized in bundles produced by Patient-everything where each bundle contains its own copy of a resource such as Practitioner that might be referred to by many patients. If some resources fail to import, for example due to parsing errors, successfully imported resources are not rolled back. The location and format of the input data are specified by the parameters in ImportResourcesRequest. Note that if no format is specified, this method assumes the `BUNDLE` format. When using the `BUNDLE` format this method ignores the `Bundle.type` field, except that `history` bundles are rejected, and does not apply any of the bundle processing semantics for batch or transaction bundles. Unlike in ExecuteBundle, transaction bundles are not executed as a single transaction and bundle-internal references are not rewritten. The bundle is treated as a collection of resources to be written as provided in `Bundle.entry.resource`, ignoring `Bundle.entry.request`. As an example, this allows the import of `searchset` bundles produced by a FHIR search or Patient-everything operation. This method returns an Operation that can be used to track the status of the import by calling GetOperation. Immediate fatal errors appear in the error field, errors are also logged to Cloud Logging (see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging)). Otherwise, when the operation finishes, a detailed response of type ImportResourcesResponse is returned in the response field. The metadata field type for this operation is OperationMetadata.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores/{fhirStoresId}:import', http_method='POST', method_id='healthcare.projects.locations.datasets.fhirStores.import', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:import', request_field='importResourcesRequest', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresImportRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the FHIR stores in the given dataset.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFhirStoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores', http_method='GET', method_id='healthcare.projects.locations.datasets.fhirStores.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/fhirStores', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresListRequest', response_type_name='ListFhirStoresResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the configuration of the specified FHIR store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FhirStore) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores/{fhirStoresId}', http_method='PATCH', method_id='healthcare.projects.locations.datasets.fhirStores.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='fhirStore', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresPatchRequest', response_type_name='FhirStore', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores/{fhirStoresId}:setIamPolicy', http_method='POST', method_id='healthcare.projects.locations.datasets.fhirStores.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/fhirStores/{fhirStoresId}:testIamPermissions', http_method='POST', method_id='healthcare.projects.locations.datasets.fhirStores.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='HealthcareProjectsLocationsDatasetsFhirStoresTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)