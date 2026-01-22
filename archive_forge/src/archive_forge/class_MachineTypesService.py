from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class MachineTypesService(base_api.BaseApiService):
    """Service class for the machineTypes resource."""
    _NAME = 'machineTypes'

    def __init__(self, client):
        super(ComputeBeta.MachineTypesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of machine types. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeMachineTypesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MachineTypeAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.machineTypes.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/machineTypes', request_field='', request_type_name='ComputeMachineTypesAggregatedListRequest', response_type_name='MachineTypeAggregatedList', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified machine type.

      Args:
        request: (ComputeMachineTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MachineType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.machineTypes.get', ordered_params=['project', 'zone', 'machineType'], path_params=['machineType', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/machineTypes/{machineType}', request_field='', request_type_name='ComputeMachineTypesGetRequest', response_type_name='MachineType', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of machine types available to the specified project.

      Args:
        request: (ComputeMachineTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MachineTypeList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.machineTypes.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/machineTypes', request_field='', request_type_name='ComputeMachineTypesListRequest', response_type_name='MachineTypeList', supports_download=False)