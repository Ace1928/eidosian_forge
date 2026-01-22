from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.deploymentmanager.v2 import deploymentmanager_v2_messages as messages
class DeploymentsService(base_api.BaseApiService):
    """Service class for the deployments resource."""
    _NAME = 'deployments'

    def __init__(self, client):
        super(DeploymentmanagerV2.DeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def CancelPreview(self, request, global_params=None):
        """Cancels and removes the preview currently associated with the deployment.

      Args:
        request: (DeploymentmanagerDeploymentsCancelPreviewRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CancelPreview')
        return self._RunMethod(config, request, global_params=global_params)
    CancelPreview.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='deploymentmanager.deployments.cancelPreview', ordered_params=['project', 'deployment'], path_params=['deployment', 'project'], query_params=[], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{deployment}/cancelPreview', request_field='deploymentsCancelPreviewRequest', request_type_name='DeploymentmanagerDeploymentsCancelPreviewRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a deployment and all of the resources in the deployment.

      Args:
        request: (DeploymentmanagerDeploymentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='deploymentmanager.deployments.delete', ordered_params=['project', 'deployment'], path_params=['deployment', 'project'], query_params=['deletePolicy'], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{deployment}', request_field='', request_type_name='DeploymentmanagerDeploymentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a specific deployment.

      Args:
        request: (DeploymentmanagerDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Deployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.deployments.get', ordered_params=['project', 'deployment'], path_params=['deployment', 'project'], query_params=[], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{deployment}', request_field='', request_type_name='DeploymentmanagerDeploymentsGetRequest', response_type_name='Deployment', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (DeploymentmanagerDeploymentsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.deployments.getIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{resource}/getIamPolicy', request_field='', request_type_name='DeploymentmanagerDeploymentsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a deployment and all of the resources described by the deployment manifest.

      Args:
        request: (DeploymentmanagerDeploymentsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='deploymentmanager.deployments.insert', ordered_params=['project'], path_params=['project'], query_params=['createPolicy', 'preview'], relative_path='deploymentmanager/v2/projects/{project}/global/deployments', request_field='deployment', request_type_name='DeploymentmanagerDeploymentsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all deployments for a given project.

      Args:
        request: (DeploymentmanagerDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeploymentsListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.deployments.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken'], relative_path='deploymentmanager/v2/projects/{project}/global/deployments', request_field='', request_type_name='DeploymentmanagerDeploymentsListRequest', response_type_name='DeploymentsListResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches a deployment and all of the resources described by the deployment manifest.

      Args:
        request: (DeploymentmanagerDeploymentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='deploymentmanager.deployments.patch', ordered_params=['project', 'deployment'], path_params=['deployment', 'project'], query_params=['createPolicy', 'deletePolicy', 'preview'], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{deployment}', request_field='deploymentResource', request_type_name='DeploymentmanagerDeploymentsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (DeploymentmanagerDeploymentsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='deploymentmanager.deployments.setIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{resource}/setIamPolicy', request_field='globalSetPolicyRequest', request_type_name='DeploymentmanagerDeploymentsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Stop(self, request, global_params=None):
        """Stops an ongoing operation. This does not roll back any work that has already been completed, but prevents any new work from being started.

      Args:
        request: (DeploymentmanagerDeploymentsStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='deploymentmanager.deployments.stop', ordered_params=['project', 'deployment'], path_params=['deployment', 'project'], query_params=[], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{deployment}/stop', request_field='deploymentsStopRequest', request_type_name='DeploymentmanagerDeploymentsStopRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (DeploymentmanagerDeploymentsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='deploymentmanager.deployments.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='DeploymentmanagerDeploymentsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a deployment and all of the resources described by the deployment manifest.

      Args:
        request: (DeploymentmanagerDeploymentsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='deploymentmanager.deployments.update', ordered_params=['project', 'deployment'], path_params=['deployment', 'project'], query_params=['createPolicy', 'deletePolicy', 'preview'], relative_path='deploymentmanager/v2/projects/{project}/global/deployments/{deployment}', request_field='deploymentResource', request_type_name='DeploymentmanagerDeploymentsUpdateRequest', response_type_name='Operation', supports_download=False)