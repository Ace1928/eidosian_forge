from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsKeystoresService(base_api.BaseApiService):
    """Service class for the organizations_environments_keystores resource."""
    _NAME = 'organizations_environments_keystores'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsKeystoresService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a keystore or truststore. - Keystore: Contains certificates and their associated keys. - Truststore: Contains trusted certificates used to validate a server's certificate. These certificates are typically self-signed certificates or certificates that are not signed by a trusted CA.

      Args:
        request: (ApigeeOrganizationsEnvironmentsKeystoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Keystore) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/keystores', http_method='POST', method_id='apigee.organizations.environments.keystores.create', ordered_params=['parent'], path_params=['parent'], query_params=['name'], relative_path='v1/{+parent}/keystores', request_field='googleCloudApigeeV1Keystore', request_type_name='ApigeeOrganizationsEnvironmentsKeystoresCreateRequest', response_type_name='GoogleCloudApigeeV1Keystore', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a keystore or truststore.

      Args:
        request: (ApigeeOrganizationsEnvironmentsKeystoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Keystore) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/keystores/{keystoresId}', http_method='DELETE', method_id='apigee.organizations.environments.keystores.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsKeystoresDeleteRequest', response_type_name='GoogleCloudApigeeV1Keystore', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a keystore or truststore.

      Args:
        request: (ApigeeOrganizationsEnvironmentsKeystoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Keystore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/keystores/{keystoresId}', http_method='GET', method_id='apigee.organizations.environments.keystores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsKeystoresGetRequest', response_type_name='GoogleCloudApigeeV1Keystore', supports_download=False)