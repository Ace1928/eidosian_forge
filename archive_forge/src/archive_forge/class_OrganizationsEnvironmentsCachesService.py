from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsCachesService(base_api.BaseApiService):
    """Service class for the organizations_environments_caches resource."""
    _NAME = 'organizations_environments_caches'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsCachesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a cache.

      Args:
        request: (ApigeeOrganizationsEnvironmentsCachesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/caches/{cachesId}', http_method='DELETE', method_id='apigee.organizations.environments.caches.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsCachesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)