from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
class ProjectsLocationsDataTaxonomiesAttributesService(base_api.BaseApiService):
    """Service class for the projects_locations_dataTaxonomies_attributes resource."""
    _NAME = 'projects_locations_dataTaxonomies_attributes'

    def __init__(self, client):
        super(DataplexV1.ProjectsLocationsDataTaxonomiesAttributesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a DataAttribute resource.

      Args:
        request: (DataplexProjectsLocationsDataTaxonomiesAttributesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataTaxonomies/{dataTaxonomiesId}/attributes', http_method='POST', method_id='dataplex.projects.locations.dataTaxonomies.attributes.create', ordered_params=['parent'], path_params=['parent'], query_params=['dataAttributeId', 'validateOnly'], relative_path='v1/{+parent}/attributes', request_field='googleCloudDataplexV1DataAttribute', request_type_name='DataplexProjectsLocationsDataTaxonomiesAttributesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Data Attribute resource.

      Args:
        request: (DataplexProjectsLocationsDataTaxonomiesAttributesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataTaxonomies/{dataTaxonomiesId}/attributes/{attributesId}', http_method='DELETE', method_id='dataplex.projects.locations.dataTaxonomies.attributes.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsDataTaxonomiesAttributesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a Data Attribute resource.

      Args:
        request: (DataplexProjectsLocationsDataTaxonomiesAttributesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1DataAttribute) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataTaxonomies/{dataTaxonomiesId}/attributes/{attributesId}', http_method='GET', method_id='dataplex.projects.locations.dataTaxonomies.attributes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsDataTaxonomiesAttributesGetRequest', response_type_name='GoogleCloudDataplexV1DataAttribute', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DataplexProjectsLocationsDataTaxonomiesAttributesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataTaxonomies/{dataTaxonomiesId}/attributes/{attributesId}:getIamPolicy', http_method='GET', method_id='dataplex.projects.locations.dataTaxonomies.attributes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='DataplexProjectsLocationsDataTaxonomiesAttributesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Data Attribute resources in a DataTaxonomy.

      Args:
        request: (DataplexProjectsLocationsDataTaxonomiesAttributesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1ListDataAttributesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataTaxonomies/{dataTaxonomiesId}/attributes', http_method='GET', method_id='dataplex.projects.locations.dataTaxonomies.attributes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/attributes', request_field='', request_type_name='DataplexProjectsLocationsDataTaxonomiesAttributesListRequest', response_type_name='GoogleCloudDataplexV1ListDataAttributesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a DataAttribute resource.

      Args:
        request: (DataplexProjectsLocationsDataTaxonomiesAttributesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataTaxonomies/{dataTaxonomiesId}/attributes/{attributesId}', http_method='PATCH', method_id='dataplex.projects.locations.dataTaxonomies.attributes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='googleCloudDataplexV1DataAttribute', request_type_name='DataplexProjectsLocationsDataTaxonomiesAttributesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.Can return NOT_FOUND, INVALID_ARGUMENT, and PERMISSION_DENIED errors.

      Args:
        request: (DataplexProjectsLocationsDataTaxonomiesAttributesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataTaxonomies/{dataTaxonomiesId}/attributes/{attributesId}:setIamPolicy', http_method='POST', method_id='dataplex.projects.locations.dataTaxonomies.attributes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='DataplexProjectsLocationsDataTaxonomiesAttributesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DataplexProjectsLocationsDataTaxonomiesAttributesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataTaxonomies/{dataTaxonomiesId}/attributes/{attributesId}:testIamPermissions', http_method='POST', method_id='dataplex.projects.locations.dataTaxonomies.attributes.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='DataplexProjectsLocationsDataTaxonomiesAttributesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)