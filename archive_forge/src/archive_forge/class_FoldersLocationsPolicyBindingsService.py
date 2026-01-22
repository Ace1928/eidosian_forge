from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v3beta import iam_v3beta_messages as messages
class FoldersLocationsPolicyBindingsService(base_api.BaseApiService):
    """Service class for the folders_locations_policyBindings resource."""
    _NAME = 'folders_locations_policyBindings'

    def __init__(self, client):
        super(IamV3beta.FoldersLocationsPolicyBindingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a policy binding and returns a long-running operation. Callers will need the IAM permissions on both the policy and target. Once the binding is created, the policy is applied to the target.

      Args:
        request: (IamFoldersLocationsPolicyBindingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/folders/{foldersId}/locations/{locationsId}/policyBindings', http_method='POST', method_id='iam.folders.locations.policyBindings.create', ordered_params=['parent'], path_params=['parent'], query_params=['policyBindingId', 'validateOnly'], relative_path='v3beta/{+parent}/policyBindings', request_field='googleIamV3betaPolicyBinding', request_type_name='IamFoldersLocationsPolicyBindingsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a policy binding. Callers will need the IAM permissions on both the policy and target. Once the binding is deleted, the policy no longer applies to the target.

      Args:
        request: (IamFoldersLocationsPolicyBindingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/folders/{foldersId}/locations/{locationsId}/policyBindings/{policyBindingsId}', http_method='DELETE', method_id='iam.folders.locations.policyBindings.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'validateOnly'], relative_path='v3beta/{+name}', request_field='', request_type_name='IamFoldersLocationsPolicyBindingsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a policy binding.

      Args:
        request: (IamFoldersLocationsPolicyBindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaPolicyBinding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/folders/{foldersId}/locations/{locationsId}/policyBindings/{policyBindingsId}', http_method='GET', method_id='iam.folders.locations.policyBindings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3beta/{+name}', request_field='', request_type_name='IamFoldersLocationsPolicyBindingsGetRequest', response_type_name='GoogleIamV3betaPolicyBinding', supports_download=False)

    def List(self, request, global_params=None):
        """Lists policy bindings.

      Args:
        request: (IamFoldersLocationsPolicyBindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaListPolicyBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/folders/{foldersId}/locations/{locationsId}/policyBindings', http_method='GET', method_id='iam.folders.locations.policyBindings.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v3beta/{+parent}/policyBindings', request_field='', request_type_name='IamFoldersLocationsPolicyBindingsListRequest', response_type_name='GoogleIamV3betaListPolicyBindingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a policy binding. Callers will need the IAM permissions on the policy and target in the binding to update, and the IAM permission to remove the existing policy from the binding. Target is immutable and cannot be updated. Once the binding is updated, the new policy is applied to the target.

      Args:
        request: (IamFoldersLocationsPolicyBindingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/folders/{foldersId}/locations/{locationsId}/policyBindings/{policyBindingsId}', http_method='PATCH', method_id='iam.folders.locations.policyBindings.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v3beta/{+name}', request_field='googleIamV3betaPolicyBinding', request_type_name='IamFoldersLocationsPolicyBindingsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SearchTargetPolicyBindings(self, request, global_params=None):
        """Search policy bindings by target. Returns all policy binding objects bound directly to target.

      Args:
        request: (IamFoldersLocationsPolicyBindingsSearchTargetPolicyBindingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaSearchTargetPolicyBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchTargetPolicyBindings')
        return self._RunMethod(config, request, global_params=global_params)
    SearchTargetPolicyBindings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/folders/{foldersId}/locations/{locationsId}/policyBindings:searchTargetPolicyBindings', http_method='GET', method_id='iam.folders.locations.policyBindings.searchTargetPolicyBindings', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'target'], relative_path='v3beta/{+parent}/policyBindings:searchTargetPolicyBindings', request_field='', request_type_name='IamFoldersLocationsPolicyBindingsSearchTargetPolicyBindingsRequest', response_type_name='GoogleIamV3betaSearchTargetPolicyBindingsResponse', supports_download=False)