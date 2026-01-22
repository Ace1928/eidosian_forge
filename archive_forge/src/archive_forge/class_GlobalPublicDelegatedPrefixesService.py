from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class GlobalPublicDelegatedPrefixesService(base_api.BaseApiService):
    """Service class for the globalPublicDelegatedPrefixes resource."""
    _NAME = 'globalPublicDelegatedPrefixes'

    def __init__(self, client):
        super(ComputeBeta.GlobalPublicDelegatedPrefixesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified global PublicDelegatedPrefix.

      Args:
        request: (ComputeGlobalPublicDelegatedPrefixesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.globalPublicDelegatedPrefixes.delete', ordered_params=['project', 'publicDelegatedPrefix'], path_params=['project', 'publicDelegatedPrefix'], query_params=['requestId'], relative_path='projects/{project}/global/publicDelegatedPrefixes/{publicDelegatedPrefix}', request_field='', request_type_name='ComputeGlobalPublicDelegatedPrefixesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified global PublicDelegatedPrefix resource.

      Args:
        request: (ComputeGlobalPublicDelegatedPrefixesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublicDelegatedPrefix) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalPublicDelegatedPrefixes.get', ordered_params=['project', 'publicDelegatedPrefix'], path_params=['project', 'publicDelegatedPrefix'], query_params=[], relative_path='projects/{project}/global/publicDelegatedPrefixes/{publicDelegatedPrefix}', request_field='', request_type_name='ComputeGlobalPublicDelegatedPrefixesGetRequest', response_type_name='PublicDelegatedPrefix', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a global PublicDelegatedPrefix in the specified project using the parameters that are included in the request.

      Args:
        request: (ComputeGlobalPublicDelegatedPrefixesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalPublicDelegatedPrefixes.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/publicDelegatedPrefixes', request_field='publicDelegatedPrefix', request_type_name='ComputeGlobalPublicDelegatedPrefixesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the global PublicDelegatedPrefixes for a project.

      Args:
        request: (ComputeGlobalPublicDelegatedPrefixesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublicDelegatedPrefixList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalPublicDelegatedPrefixes.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/publicDelegatedPrefixes', request_field='', request_type_name='ComputeGlobalPublicDelegatedPrefixesListRequest', response_type_name='PublicDelegatedPrefixList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified global PublicDelegatedPrefix resource with the data included in the request. This method supports PATCH semantics and uses JSON merge patch format and processing rules.

      Args:
        request: (ComputeGlobalPublicDelegatedPrefixesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.globalPublicDelegatedPrefixes.patch', ordered_params=['project', 'publicDelegatedPrefix'], path_params=['project', 'publicDelegatedPrefix'], query_params=['requestId'], relative_path='projects/{project}/global/publicDelegatedPrefixes/{publicDelegatedPrefix}', request_field='publicDelegatedPrefixResource', request_type_name='ComputeGlobalPublicDelegatedPrefixesPatchRequest', response_type_name='Operation', supports_download=False)