from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsReferencesService(base_api.BaseApiService):
    """Service class for the organizations_environments_references resource."""
    _NAME = 'organizations_environments_references'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsReferencesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Reference in the specified environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsReferencesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Reference) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/references', http_method='POST', method_id='apigee.organizations.environments.references.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/references', request_field='googleCloudApigeeV1Reference', request_type_name='ApigeeOrganizationsEnvironmentsReferencesCreateRequest', response_type_name='GoogleCloudApigeeV1Reference', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Reference from an environment. Returns the deleted Reference resource.

      Args:
        request: (ApigeeOrganizationsEnvironmentsReferencesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Reference) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/references/{referencesId}', http_method='DELETE', method_id='apigee.organizations.environments.references.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsReferencesDeleteRequest', response_type_name='GoogleCloudApigeeV1Reference', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Reference resource.

      Args:
        request: (ApigeeOrganizationsEnvironmentsReferencesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Reference) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/references/{referencesId}', http_method='GET', method_id='apigee.organizations.environments.references.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsReferencesGetRequest', response_type_name='GoogleCloudApigeeV1Reference', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing Reference. Note that this operation has PUT semantics; it will replace the entirety of the existing Reference with the resource in the request body.

      Args:
        request: (GoogleCloudApigeeV1Reference) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Reference) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/references/{referencesId}', http_method='PUT', method_id='apigee.organizations.environments.references.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1Reference', response_type_name='GoogleCloudApigeeV1Reference', supports_download=False)