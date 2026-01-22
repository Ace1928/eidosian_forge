from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
class ProjectsLocationsTaxonomiesPolicyTagsService(base_api.BaseApiService):
    """Service class for the projects_locations_taxonomies_policyTags resource."""
    _NAME = 'projects_locations_taxonomies_policyTags'

    def __init__(self, client):
        super(DatacatalogV1.ProjectsLocationsTaxonomiesPolicyTagsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a policy tag in a taxonomy.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesPolicyTagsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1PolicyTag) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}/policyTags', http_method='POST', method_id='datacatalog.projects.locations.taxonomies.policyTags.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/policyTags', request_field='googleCloudDatacatalogV1PolicyTag', request_type_name='DatacatalogProjectsLocationsTaxonomiesPolicyTagsCreateRequest', response_type_name='GoogleCloudDatacatalogV1PolicyTag', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a policy tag together with the following: * All of its descendant policy tags, if any * Policies associated with the policy tag and its descendants * References from BigQuery table schema of the policy tag and its descendants.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesPolicyTagsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}/policyTags/{policyTagsId}', http_method='DELETE', method_id='datacatalog.projects.locations.taxonomies.policyTags.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatacatalogProjectsLocationsTaxonomiesPolicyTagsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a policy tag.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesPolicyTagsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1PolicyTag) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}/policyTags/{policyTagsId}', http_method='GET', method_id='datacatalog.projects.locations.taxonomies.policyTags.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatacatalogProjectsLocationsTaxonomiesPolicyTagsGetRequest', response_type_name='GoogleCloudDatacatalogV1PolicyTag', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM policy for a policy tag or a taxonomy.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesPolicyTagsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}/policyTags/{policyTagsId}:getIamPolicy', http_method='POST', method_id='datacatalog.projects.locations.taxonomies.policyTags.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='DatacatalogProjectsLocationsTaxonomiesPolicyTagsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all policy tags in a taxonomy.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesPolicyTagsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1ListPolicyTagsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}/policyTags', http_method='GET', method_id='datacatalog.projects.locations.taxonomies.policyTags.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/policyTags', request_field='', request_type_name='DatacatalogProjectsLocationsTaxonomiesPolicyTagsListRequest', response_type_name='GoogleCloudDatacatalogV1ListPolicyTagsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a policy tag, including its display name, description, and parent policy tag.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesPolicyTagsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1PolicyTag) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}/policyTags/{policyTagsId}', http_method='PATCH', method_id='datacatalog.projects.locations.taxonomies.policyTags.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudDatacatalogV1PolicyTag', request_type_name='DatacatalogProjectsLocationsTaxonomiesPolicyTagsPatchRequest', response_type_name='GoogleCloudDatacatalogV1PolicyTag', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM policy for a policy tag or a taxonomy.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesPolicyTagsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}/policyTags/{policyTagsId}:setIamPolicy', http_method='POST', method_id='datacatalog.projects.locations.taxonomies.policyTags.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DatacatalogProjectsLocationsTaxonomiesPolicyTagsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns your permissions on a specified policy tag or taxonomy.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesPolicyTagsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/taxonomies/{taxonomiesId}/policyTags/{policyTagsId}:testIamPermissions', http_method='POST', method_id='datacatalog.projects.locations.taxonomies.policyTags.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DatacatalogProjectsLocationsTaxonomiesPolicyTagsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)