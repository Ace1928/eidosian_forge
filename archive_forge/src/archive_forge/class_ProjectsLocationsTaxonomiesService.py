from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
class ProjectsLocationsTaxonomiesService(base_api.BaseApiService):
    """Service class for the projects_locations_taxonomies resource."""
    _NAME = 'projects_locations_taxonomies'

    def __init__(self, client):
        super(DatacatalogV1.ProjectsLocationsTaxonomiesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a taxonomy in a specified project. The taxonomy is initially empty, that is, it doesn't contain policy tags.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Taxonomy) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies', http_method='POST', method_id='datacatalog.projects.locations.taxonomies.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/taxonomies', request_field='googleCloudDatacatalogV1Taxonomy', request_type_name='DatacatalogProjectsLocationsTaxonomiesCreateRequest', response_type_name='GoogleCloudDatacatalogV1Taxonomy', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a taxonomy, including all policy tags in this taxonomy, their associated policies, and the policy tags references from BigQuery columns.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}', http_method='DELETE', method_id='datacatalog.projects.locations.taxonomies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatacatalogProjectsLocationsTaxonomiesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Export(self, request, global_params=None):
        """Exports taxonomies in the requested type and returns them, including their policy tags. The requested taxonomies must belong to the same project. This method generates `SerializedTaxonomy` protocol buffers with nested policy tags that can be used as input for `ImportTaxonomies` calls.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1ExportTaxonomiesResponse) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies:export', http_method='GET', method_id='datacatalog.projects.locations.taxonomies.export', ordered_params=['parent'], path_params=['parent'], query_params=['serializedTaxonomies', 'taxonomies'], relative_path='v1/{+parent}/taxonomies:export', request_field='', request_type_name='DatacatalogProjectsLocationsTaxonomiesExportRequest', response_type_name='GoogleCloudDatacatalogV1ExportTaxonomiesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a taxonomy.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Taxonomy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}', http_method='GET', method_id='datacatalog.projects.locations.taxonomies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatacatalogProjectsLocationsTaxonomiesGetRequest', response_type_name='GoogleCloudDatacatalogV1Taxonomy', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM policy for a policy tag or a taxonomy.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}:getIamPolicy', http_method='POST', method_id='datacatalog.projects.locations.taxonomies.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='DatacatalogProjectsLocationsTaxonomiesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Import(self, request, global_params=None):
        """Creates new taxonomies (including their policy tags) in a given project by importing from inlined or cross-regional sources. For a cross-regional source, new taxonomies are created by copying from a source in another region. For an inlined source, taxonomies and policy tags are created in bulk using nested protocol buffer structures.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1ImportTaxonomiesResponse) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies:import', http_method='POST', method_id='datacatalog.projects.locations.taxonomies.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/taxonomies:import', request_field='googleCloudDatacatalogV1ImportTaxonomiesRequest', request_type_name='DatacatalogProjectsLocationsTaxonomiesImportRequest', response_type_name='GoogleCloudDatacatalogV1ImportTaxonomiesResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all taxonomies in a project in a particular location that you have a permission to view.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1ListTaxonomiesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies', http_method='GET', method_id='datacatalog.projects.locations.taxonomies.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/taxonomies', request_field='', request_type_name='DatacatalogProjectsLocationsTaxonomiesListRequest', response_type_name='GoogleCloudDatacatalogV1ListTaxonomiesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a taxonomy, including its display name, description, and activated policy types.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Taxonomy) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}', http_method='PATCH', method_id='datacatalog.projects.locations.taxonomies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudDatacatalogV1Taxonomy', request_type_name='DatacatalogProjectsLocationsTaxonomiesPatchRequest', response_type_name='GoogleCloudDatacatalogV1Taxonomy', supports_download=False)

    def Replace(self, request, global_params=None):
        """Replaces (updates) a taxonomy and all its policy tags. The taxonomy and its entire hierarchy of policy tags must be represented literally by `SerializedTaxonomy` and the nested `SerializedPolicyTag` messages. This operation automatically does the following: - Deletes the existing policy tags that are missing from the `SerializedPolicyTag`. - Creates policy tags that don't have resource names. They are considered new. - Updates policy tags with valid resources names accordingly.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesReplaceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Taxonomy) The response message.
      """
        config = self.GetMethodConfig('Replace')
        return self._RunMethod(config, request, global_params=global_params)
    Replace.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}:replace', http_method='POST', method_id='datacatalog.projects.locations.taxonomies.replace', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:replace', request_field='googleCloudDatacatalogV1ReplaceTaxonomyRequest', request_type_name='DatacatalogProjectsLocationsTaxonomiesReplaceRequest', response_type_name='GoogleCloudDatacatalogV1Taxonomy', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM policy for a policy tag or a taxonomy.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}:setIamPolicy', http_method='POST', method_id='datacatalog.projects.locations.taxonomies.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DatacatalogProjectsLocationsTaxonomiesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns your permissions on a specified policy tag or taxonomy.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}:testIamPermissions', http_method='POST', method_id='datacatalog.projects.locations.taxonomies.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DatacatalogProjectsLocationsTaxonomiesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)