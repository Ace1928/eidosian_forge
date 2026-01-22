from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class GlobalAddressesService(base_api.BaseApiService):
    """Service class for the globalAddresses resource."""
    _NAME = 'globalAddresses'

    def __init__(self, client):
        super(ComputeBeta.GlobalAddressesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified address resource.

      Args:
        request: (ComputeGlobalAddressesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.globalAddresses.delete', ordered_params=['project', 'address'], path_params=['address', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/addresses/{address}', request_field='', request_type_name='ComputeGlobalAddressesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified address resource.

      Args:
        request: (ComputeGlobalAddressesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Address) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalAddresses.get', ordered_params=['project', 'address'], path_params=['address', 'project'], query_params=[], relative_path='projects/{project}/global/addresses/{address}', request_field='', request_type_name='ComputeGlobalAddressesGetRequest', response_type_name='Address', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates an address resource in the specified project by using the data included in the request.

      Args:
        request: (ComputeGlobalAddressesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalAddresses.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/addresses', request_field='address', request_type_name='ComputeGlobalAddressesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of global addresses.

      Args:
        request: (ComputeGlobalAddressesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AddressList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalAddresses.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/addresses', request_field='', request_type_name='ComputeGlobalAddressesListRequest', response_type_name='AddressList', supports_download=False)

    def Move(self, request, global_params=None):
        """Moves the specified address resource from one project to another project.

      Args:
        request: (ComputeGlobalAddressesMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Move')
        return self._RunMethod(config, request, global_params=global_params)
    Move.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalAddresses.move', ordered_params=['project', 'address'], path_params=['address', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/addresses/{address}/move', request_field='globalAddressesMoveRequest', request_type_name='ComputeGlobalAddressesMoveRequest', response_type_name='Operation', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on a GlobalAddress. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeGlobalAddressesSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalAddresses.setLabels', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/addresses/{resource}/setLabels', request_field='globalSetLabelsRequest', request_type_name='ComputeGlobalAddressesSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeGlobalAddressesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalAddresses.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/addresses/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeGlobalAddressesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)