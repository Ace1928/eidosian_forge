from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v2alpha import serviceusage_v2alpha_messages as messages
class ServicesGroupsDescendantServicesService(base_api.BaseApiService):
    """Service class for the services_groups_descendantServices resource."""
    _NAME = 'services_groups_descendantServices'

    def __init__(self, client):
        super(ServiceusageV2alpha.ServicesGroupsDescendantServicesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List the services that belong to a given service group or transitively to any of the groups that are members of the service group. The service group is a producer defined service group.

      Args:
        request: (ServiceusageServicesGroupsDescendantServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDescendantServicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/{v2alphaId}/{v2alphaId1}/services/{servicesId}/groups/{groupsId}/descendantServices', http_method='GET', method_id='serviceusage.services.groups.descendantServices.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2alpha/{+parent}/descendantServices', request_field='', request_type_name='ServiceusageServicesGroupsDescendantServicesListRequest', response_type_name='ListDescendantServicesResponse', supports_download=False)