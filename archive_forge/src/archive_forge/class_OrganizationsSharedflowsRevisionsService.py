from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsSharedflowsRevisionsService(base_api.BaseApiService):
    """Service class for the organizations_sharedflows_revisions resource."""
    _NAME = 'organizations_sharedflows_revisions'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsSharedflowsRevisionsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a shared flow and all associated policies, resources, and revisions. You must undeploy the shared flow before deleting it.

      Args:
        request: (ApigeeOrganizationsSharedflowsRevisionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SharedFlowRevision) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sharedflows/{sharedflowsId}/revisions/{revisionsId}', http_method='DELETE', method_id='apigee.organizations.sharedflows.revisions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSharedflowsRevisionsDeleteRequest', response_type_name='GoogleCloudApigeeV1SharedFlowRevision', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a revision of a shared flow. To download the shared flow configuration bundle for the specified revision as a zip file, set the `format` query parameter to `bundle`. If you are using curl, specify `-o filename.zip` to save the output to a file; otherwise, it displays to `stdout`. Then, develop the shared flow configuration locally and upload the updated sharedFlow configuration revision, as described in [updateSharedFlowRevision](updateSharedFlowRevision).

      Args:
        request: (ApigeeOrganizationsSharedflowsRevisionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sharedflows/{sharedflowsId}/revisions/{revisionsId}', http_method='GET', method_id='apigee.organizations.sharedflows.revisions.get', ordered_params=['name'], path_params=['name'], query_params=['format'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSharedflowsRevisionsGetRequest', response_type_name='GoogleApiHttpBody', supports_download=False)

    def UpdateSharedFlowRevision(self, request, global_params=None):
        """Updates a shared flow revision. This operation is only allowed on revisions which have never been deployed. After deployment a revision becomes immutable, even if it becomes undeployed. The payload is a ZIP-formatted shared flow. Content type must be either multipart/form-data or application/octet-stream.

      Args:
        request: (ApigeeOrganizationsSharedflowsRevisionsUpdateSharedFlowRevisionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SharedFlowRevision) The response message.
      """
        config = self.GetMethodConfig('UpdateSharedFlowRevision')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateSharedFlowRevision.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sharedflows/{sharedflowsId}/revisions/{revisionsId}', http_method='POST', method_id='apigee.organizations.sharedflows.revisions.updateSharedFlowRevision', ordered_params=['name'], path_params=['name'], query_params=['validate'], relative_path='v1/{+name}', request_field='googleApiHttpBody', request_type_name='ApigeeOrganizationsSharedflowsRevisionsUpdateSharedFlowRevisionRequest', response_type_name='GoogleCloudApigeeV1SharedFlowRevision', supports_download=False)