from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages as messages
class OrganizationsCustomConstraintsService(base_api.BaseApiService):
    """Service class for the organizations_customConstraints resource."""
    _NAME = 'organizations_customConstraints'

    def __init__(self, client):
        super(OrgpolicyV2.OrganizationsCustomConstraintsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a custom constraint. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the organization does not exist. Returns a `google.rpc.Status` with `google.rpc.Code.ALREADY_EXISTS` if the constraint already exists on the given organization.

      Args:
        request: (OrgpolicyOrganizationsCustomConstraintsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2CustomConstraint) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/customConstraints', http_method='POST', method_id='orgpolicy.organizations.customConstraints.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/customConstraints', request_field='googleCloudOrgpolicyV2CustomConstraint', request_type_name='OrgpolicyOrganizationsCustomConstraintsCreateRequest', response_type_name='GoogleCloudOrgpolicyV2CustomConstraint', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a custom constraint. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the constraint does not exist.

      Args:
        request: (OrgpolicyOrganizationsCustomConstraintsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/customConstraints/{customConstraintsId}', http_method='DELETE', method_id='orgpolicy.organizations.customConstraints.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='OrgpolicyOrganizationsCustomConstraintsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a custom constraint. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the custom constraint does not exist.

      Args:
        request: (OrgpolicyOrganizationsCustomConstraintsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2CustomConstraint) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/customConstraints/{customConstraintsId}', http_method='GET', method_id='orgpolicy.organizations.customConstraints.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='OrgpolicyOrganizationsCustomConstraintsGetRequest', response_type_name='GoogleCloudOrgpolicyV2CustomConstraint', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves all of the custom constraints that exist on a particular organization resource.

      Args:
        request: (OrgpolicyOrganizationsCustomConstraintsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2ListCustomConstraintsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/customConstraints', http_method='GET', method_id='orgpolicy.organizations.customConstraints.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/customConstraints', request_field='', request_type_name='OrgpolicyOrganizationsCustomConstraintsListRequest', response_type_name='GoogleCloudOrgpolicyV2ListCustomConstraintsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a custom constraint. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the constraint does not exist. Note: the supplied policy will perform a full overwrite of all fields.

      Args:
        request: (OrgpolicyOrganizationsCustomConstraintsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2CustomConstraint) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/customConstraints/{customConstraintsId}', http_method='PATCH', method_id='orgpolicy.organizations.customConstraints.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='googleCloudOrgpolicyV2CustomConstraint', request_type_name='OrgpolicyOrganizationsCustomConstraintsPatchRequest', response_type_name='GoogleCloudOrgpolicyV2CustomConstraint', supports_download=False)