from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class InstanceTemplatesService(base_api.BaseApiService):
    """Service class for the instanceTemplates resource."""
    _NAME = 'instanceTemplates'

    def __init__(self, client):
        super(ComputeBeta.InstanceTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all InstanceTemplates resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeInstanceTemplatesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceTemplateAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceTemplates.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/instanceTemplates', request_field='', request_type_name='ComputeInstanceTemplatesAggregatedListRequest', response_type_name='InstanceTemplateAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified instance template. Deleting an instance template is permanent and cannot be undone. It is not possible to delete templates that are already in use by a managed instance group.

      Args:
        request: (ComputeInstanceTemplatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.instanceTemplates.delete', ordered_params=['project', 'instanceTemplate'], path_params=['instanceTemplate', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/instanceTemplates/{instanceTemplate}', request_field='', request_type_name='ComputeInstanceTemplatesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified instance template.

      Args:
        request: (ComputeInstanceTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceTemplates.get', ordered_params=['project', 'instanceTemplate'], path_params=['instanceTemplate', 'project'], query_params=['view'], relative_path='projects/{project}/global/instanceTemplates/{instanceTemplate}', request_field='', request_type_name='ComputeInstanceTemplatesGetRequest', response_type_name='InstanceTemplate', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeInstanceTemplatesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceTemplates.getIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/global/instanceTemplates/{resource}/getIamPolicy', request_field='', request_type_name='ComputeInstanceTemplatesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates an instance template in the specified project using the data that is included in the request. If you are creating a new template to update an existing instance group, your new instance template must use the same network or, if applicable, the same subnetwork as the original template.

      Args:
        request: (ComputeInstanceTemplatesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceTemplates.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/instanceTemplates', request_field='instanceTemplate', request_type_name='ComputeInstanceTemplatesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of instance templates that are contained within the specified project.

      Args:
        request: (ComputeInstanceTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceTemplateList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceTemplates.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'view'], relative_path='projects/{project}/global/instanceTemplates', request_field='', request_type_name='ComputeInstanceTemplatesListRequest', response_type_name='InstanceTemplateList', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeInstanceTemplatesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceTemplates.setIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/instanceTemplates/{resource}/setIamPolicy', request_field='globalSetPolicyRequest', request_type_name='ComputeInstanceTemplatesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeInstanceTemplatesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceTemplates.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/instanceTemplates/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeInstanceTemplatesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)