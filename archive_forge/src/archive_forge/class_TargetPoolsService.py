from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class TargetPoolsService(base_api.BaseApiService):
    """Service class for the targetPools resource."""
    _NAME = 'targetPools'

    def __init__(self, client):
        super(ComputeBeta.TargetPoolsService, self).__init__(client)
        self._upload_configs = {}

    def AddHealthCheck(self, request, global_params=None):
        """Adds health check URLs to a target pool.

      Args:
        request: (ComputeTargetPoolsAddHealthCheckRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddHealthCheck')
        return self._RunMethod(config, request, global_params=global_params)
    AddHealthCheck.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetPools.addHealthCheck', ordered_params=['project', 'region', 'targetPool'], path_params=['project', 'region', 'targetPool'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetPools/{targetPool}/addHealthCheck', request_field='targetPoolsAddHealthCheckRequest', request_type_name='ComputeTargetPoolsAddHealthCheckRequest', response_type_name='Operation', supports_download=False)

    def AddInstance(self, request, global_params=None):
        """Adds an instance to a target pool.

      Args:
        request: (ComputeTargetPoolsAddInstanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddInstance')
        return self._RunMethod(config, request, global_params=global_params)
    AddInstance.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetPools.addInstance', ordered_params=['project', 'region', 'targetPool'], path_params=['project', 'region', 'targetPool'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetPools/{targetPool}/addInstance', request_field='targetPoolsAddInstanceRequest', request_type_name='ComputeTargetPoolsAddInstanceRequest', response_type_name='Operation', supports_download=False)

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of target pools. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeTargetPoolsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetPoolAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetPools.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/targetPools', request_field='', request_type_name='ComputeTargetPoolsAggregatedListRequest', response_type_name='TargetPoolAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified target pool.

      Args:
        request: (ComputeTargetPoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.targetPools.delete', ordered_params=['project', 'region', 'targetPool'], path_params=['project', 'region', 'targetPool'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetPools/{targetPool}', request_field='', request_type_name='ComputeTargetPoolsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified target pool.

      Args:
        request: (ComputeTargetPoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetPool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetPools.get', ordered_params=['project', 'region', 'targetPool'], path_params=['project', 'region', 'targetPool'], query_params=[], relative_path='projects/{project}/regions/{region}/targetPools/{targetPool}', request_field='', request_type_name='ComputeTargetPoolsGetRequest', response_type_name='TargetPool', supports_download=False)

    def GetHealth(self, request, global_params=None):
        """Gets the most recent health check results for each IP for the instance that is referenced by the given target pool.

      Args:
        request: (ComputeTargetPoolsGetHealthRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetPoolInstanceHealth) The response message.
      """
        config = self.GetMethodConfig('GetHealth')
        return self._RunMethod(config, request, global_params=global_params)
    GetHealth.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetPools.getHealth', ordered_params=['project', 'region', 'targetPool'], path_params=['project', 'region', 'targetPool'], query_params=[], relative_path='projects/{project}/regions/{region}/targetPools/{targetPool}/getHealth', request_field='instanceReference', request_type_name='ComputeTargetPoolsGetHealthRequest', response_type_name='TargetPoolInstanceHealth', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a target pool in the specified project and region using the data included in the request.

      Args:
        request: (ComputeTargetPoolsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetPools.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetPools', request_field='targetPool', request_type_name='ComputeTargetPoolsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of target pools available to the specified project and region.

      Args:
        request: (ComputeTargetPoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetPoolList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetPools.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/targetPools', request_field='', request_type_name='ComputeTargetPoolsListRequest', response_type_name='TargetPoolList', supports_download=False)

    def RemoveHealthCheck(self, request, global_params=None):
        """Removes health check URL from a target pool.

      Args:
        request: (ComputeTargetPoolsRemoveHealthCheckRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveHealthCheck')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveHealthCheck.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetPools.removeHealthCheck', ordered_params=['project', 'region', 'targetPool'], path_params=['project', 'region', 'targetPool'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetPools/{targetPool}/removeHealthCheck', request_field='targetPoolsRemoveHealthCheckRequest', request_type_name='ComputeTargetPoolsRemoveHealthCheckRequest', response_type_name='Operation', supports_download=False)

    def RemoveInstance(self, request, global_params=None):
        """Removes instance URL from a target pool.

      Args:
        request: (ComputeTargetPoolsRemoveInstanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveInstance')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveInstance.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetPools.removeInstance', ordered_params=['project', 'region', 'targetPool'], path_params=['project', 'region', 'targetPool'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetPools/{targetPool}/removeInstance', request_field='targetPoolsRemoveInstanceRequest', request_type_name='ComputeTargetPoolsRemoveInstanceRequest', response_type_name='Operation', supports_download=False)

    def SetBackup(self, request, global_params=None):
        """Changes a backup target pool's configurations.

      Args:
        request: (ComputeTargetPoolsSetBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetBackup')
        return self._RunMethod(config, request, global_params=global_params)
    SetBackup.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetPools.setBackup', ordered_params=['project', 'region', 'targetPool'], path_params=['project', 'region', 'targetPool'], query_params=['failoverRatio', 'requestId'], relative_path='projects/{project}/regions/{region}/targetPools/{targetPool}/setBackup', request_field='targetReference', request_type_name='ComputeTargetPoolsSetBackupRequest', response_type_name='Operation', supports_download=False)

    def SetSecurityPolicy(self, request, global_params=None):
        """Sets the Google Cloud Armor security policy for the specified target pool. For more information, see Google Cloud Armor Overview.

      Args:
        request: (ComputeTargetPoolsSetSecurityPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetSecurityPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetSecurityPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetPools.setSecurityPolicy', ordered_params=['project', 'region', 'targetPool'], path_params=['project', 'region', 'targetPool'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetPools/{targetPool}/setSecurityPolicy', request_field='securityPolicyReference', request_type_name='ComputeTargetPoolsSetSecurityPolicyRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeTargetPoolsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetPools.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/targetPools/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeTargetPoolsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)