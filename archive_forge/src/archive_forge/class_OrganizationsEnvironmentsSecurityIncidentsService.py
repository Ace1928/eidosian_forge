from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsSecurityIncidentsService(base_api.BaseApiService):
    """Service class for the organizations_environments_securityIncidents resource."""
    _NAME = 'organizations_environments_securityIncidents'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsSecurityIncidentsService, self).__init__(client)
        self._upload_configs = {}

    def BatchUpdate(self, request, global_params=None):
        """BatchUpdateSecurityIncident updates multiple existing security incidents.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityIncidentsBatchUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1BatchUpdateSecurityIncidentsResponse) The response message.
      """
        config = self.GetMethodConfig('BatchUpdate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchUpdate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityIncidents:batchUpdate', http_method='POST', method_id='apigee.organizations.environments.securityIncidents.batchUpdate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/securityIncidents:batchUpdate', request_field='googleCloudApigeeV1BatchUpdateSecurityIncidentsRequest', request_type_name='ApigeeOrganizationsEnvironmentsSecurityIncidentsBatchUpdateRequest', response_type_name='GoogleCloudApigeeV1BatchUpdateSecurityIncidentsResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """GetSecurityIncident gets the specified security incident. Returns NOT_FOUND if security incident is not present for the specified organization and environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityIncidentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityIncident) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityIncidents/{securityIncidentsId}', http_method='GET', method_id='apigee.organizations.environments.securityIncidents.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSecurityIncidentsGetRequest', response_type_name='GoogleCloudApigeeV1SecurityIncident', supports_download=False)

    def List(self, request, global_params=None):
        """ListSecurityIncidents lists all the security incident associated with the environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityIncidentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListSecurityIncidentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityIncidents', http_method='GET', method_id='apigee.organizations.environments.securityIncidents.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/securityIncidents', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSecurityIncidentsListRequest', response_type_name='GoogleCloudApigeeV1ListSecurityIncidentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """UpdateSecurityIncidents updates an existing security incident.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityIncidentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityIncident) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityIncidents/{securityIncidentsId}', http_method='PATCH', method_id='apigee.organizations.environments.securityIncidents.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1SecurityIncident', request_type_name='ApigeeOrganizationsEnvironmentsSecurityIncidentsPatchRequest', response_type_name='GoogleCloudApigeeV1SecurityIncident', supports_download=False)