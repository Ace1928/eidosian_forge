from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsSecurityProfilesEnvironmentsService(base_api.BaseApiService):
    """Service class for the organizations_securityProfiles_environments resource."""
    _NAME = 'organizations_securityProfiles_environments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsSecurityProfilesEnvironmentsService, self).__init__(client)
        self._upload_configs = {}

    def ComputeEnvironmentScores(self, request, global_params=None):
        """ComputeEnvironmentScores calculates scores for requested time range for the specified security profile and environment.

      Args:
        request: (ApigeeOrganizationsSecurityProfilesEnvironmentsComputeEnvironmentScoresRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ComputeEnvironmentScoresResponse) The response message.
      """
        config = self.GetMethodConfig('ComputeEnvironmentScores')
        return self._RunMethod(config, request, global_params=global_params)
    ComputeEnvironmentScores.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityProfiles/{securityProfilesId}/environments/{environmentsId}:computeEnvironmentScores', http_method='POST', method_id='apigee.organizations.securityProfiles.environments.computeEnvironmentScores', ordered_params=['profileEnvironment'], path_params=['profileEnvironment'], query_params=[], relative_path='v1/{+profileEnvironment}:computeEnvironmentScores', request_field='googleCloudApigeeV1ComputeEnvironmentScoresRequest', request_type_name='ApigeeOrganizationsSecurityProfilesEnvironmentsComputeEnvironmentScoresRequest', response_type_name='GoogleCloudApigeeV1ComputeEnvironmentScoresResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """CreateSecurityProfileEnvironmentAssociation creates profile environment association i.e. attaches environment to security profile.

      Args:
        request: (ApigeeOrganizationsSecurityProfilesEnvironmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityProfileEnvironmentAssociation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityProfiles/{securityProfilesId}/environments', http_method='POST', method_id='apigee.organizations.securityProfiles.environments.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/environments', request_field='googleCloudApigeeV1SecurityProfileEnvironmentAssociation', request_type_name='ApigeeOrganizationsSecurityProfilesEnvironmentsCreateRequest', response_type_name='GoogleCloudApigeeV1SecurityProfileEnvironmentAssociation', supports_download=False)

    def Delete(self, request, global_params=None):
        """DeleteSecurityProfileEnvironmentAssociation removes profile environment association i.e. detaches environment from security profile.

      Args:
        request: (ApigeeOrganizationsSecurityProfilesEnvironmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityProfiles/{securityProfilesId}/environments/{environmentsId}', http_method='DELETE', method_id='apigee.organizations.securityProfiles.environments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSecurityProfilesEnvironmentsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)