from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
class InboundSsoAssignmentsService(base_api.BaseApiService):
    """Service class for the inboundSsoAssignments resource."""
    _NAME = 'inboundSsoAssignments'

    def __init__(self, client):
        super(CloudidentityV1.InboundSsoAssignmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an InboundSsoAssignment for users and devices in a `Customer` under a given `Group` or `OrgUnit`.

      Args:
        request: (InboundSsoAssignment) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudidentity.inboundSsoAssignments.create', ordered_params=[], path_params=[], query_params=[], relative_path='v1/inboundSsoAssignments', request_field='<request>', request_type_name='InboundSsoAssignment', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an InboundSsoAssignment. To disable SSO, Create (or Update) an assignment that has `sso_mode` == `SSO_OFF`.

      Args:
        request: (CloudidentityInboundSsoAssignmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/inboundSsoAssignments/{inboundSsoAssignmentsId}', http_method='DELETE', method_id='cloudidentity.inboundSsoAssignments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityInboundSsoAssignmentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an InboundSsoAssignment.

      Args:
        request: (CloudidentityInboundSsoAssignmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InboundSsoAssignment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/inboundSsoAssignments/{inboundSsoAssignmentsId}', http_method='GET', method_id='cloudidentity.inboundSsoAssignments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityInboundSsoAssignmentsGetRequest', response_type_name='InboundSsoAssignment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the InboundSsoAssignments for a `Customer`.

      Args:
        request: (CloudidentityInboundSsoAssignmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInboundSsoAssignmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudidentity.inboundSsoAssignments.list', ordered_params=[], path_params=[], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/inboundSsoAssignments', request_field='', request_type_name='CloudidentityInboundSsoAssignmentsListRequest', response_type_name='ListInboundSsoAssignmentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an InboundSsoAssignment. The body of this request is the `inbound_sso_assignment` field and the `update_mask` is relative to that. For example: a PATCH to `/v1/inboundSsoAssignments/0abcdefg1234567&update_mask=rank` with a body of `{ "rank": 1 }` moves that (presumably group-targeted) SSO assignment to the highest priority and shifts any other group-targeted assignments down in priority.

      Args:
        request: (CloudidentityInboundSsoAssignmentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/inboundSsoAssignments/{inboundSsoAssignmentsId}', http_method='PATCH', method_id='cloudidentity.inboundSsoAssignments.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='inboundSsoAssignment', request_type_name='CloudidentityInboundSsoAssignmentsPatchRequest', response_type_name='Operation', supports_download=False)