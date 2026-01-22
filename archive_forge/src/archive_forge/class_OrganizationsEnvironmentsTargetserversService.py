from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsTargetserversService(base_api.BaseApiService):
    """Service class for the organizations_environments_targetservers resource."""
    _NAME = 'organizations_environments_targetservers'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsTargetserversService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a TargetServer in the specified environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsTargetserversCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TargetServer) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/targetservers', http_method='POST', method_id='apigee.organizations.environments.targetservers.create', ordered_params=['parent'], path_params=['parent'], query_params=['name'], relative_path='v1/{+parent}/targetservers', request_field='googleCloudApigeeV1TargetServer', request_type_name='ApigeeOrganizationsEnvironmentsTargetserversCreateRequest', response_type_name='GoogleCloudApigeeV1TargetServer', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a TargetServer from an environment. Returns the deleted TargetServer resource.

      Args:
        request: (ApigeeOrganizationsEnvironmentsTargetserversDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TargetServer) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/targetservers/{targetserversId}', http_method='DELETE', method_id='apigee.organizations.environments.targetservers.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsTargetserversDeleteRequest', response_type_name='GoogleCloudApigeeV1TargetServer', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a TargetServer resource.

      Args:
        request: (ApigeeOrganizationsEnvironmentsTargetserversGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TargetServer) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/targetservers/{targetserversId}', http_method='GET', method_id='apigee.organizations.environments.targetservers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsTargetserversGetRequest', response_type_name='GoogleCloudApigeeV1TargetServer', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing TargetServer. Note that this operation has PUT semantics; it will replace the entirety of the existing TargetServer with the resource in the request body.

      Args:
        request: (GoogleCloudApigeeV1TargetServer) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TargetServer) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/targetservers/{targetserversId}', http_method='PUT', method_id='apigee.organizations.environments.targetservers.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1TargetServer', response_type_name='GoogleCloudApigeeV1TargetServer', supports_download=False)