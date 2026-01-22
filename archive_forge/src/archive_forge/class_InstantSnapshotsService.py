from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class InstantSnapshotsService(base_api.BaseApiService):
    """Service class for the instantSnapshots resource."""
    _NAME = 'instantSnapshots'

    def __init__(self, client):
        super(ComputeBeta.InstantSnapshotsService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of instantSnapshots. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeInstantSnapshotsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstantSnapshotAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instantSnapshots.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/instantSnapshots', request_field='', request_type_name='ComputeInstantSnapshotsAggregatedListRequest', response_type_name='InstantSnapshotAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified InstantSnapshot resource. Keep in mind that deleting a single instantSnapshot might not necessarily delete all the data on that instantSnapshot. If any data on the instantSnapshot that is marked for deletion is needed for subsequent instantSnapshots, the data will be moved to the next corresponding instantSnapshot. For more information, see Deleting instantSnapshots.

      Args:
        request: (ComputeInstantSnapshotsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.instantSnapshots.delete', ordered_params=['project', 'zone', 'instantSnapshot'], path_params=['instantSnapshot', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instantSnapshots/{instantSnapshot}', request_field='', request_type_name='ComputeInstantSnapshotsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified InstantSnapshot resource in the specified zone.

      Args:
        request: (ComputeInstantSnapshotsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstantSnapshot) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instantSnapshots.get', ordered_params=['project', 'zone', 'instantSnapshot'], path_params=['instantSnapshot', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/instantSnapshots/{instantSnapshot}', request_field='', request_type_name='ComputeInstantSnapshotsGetRequest', response_type_name='InstantSnapshot', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeInstantSnapshotsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instantSnapshots.getIamPolicy', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/zones/{zone}/instantSnapshots/{resource}/getIamPolicy', request_field='', request_type_name='ComputeInstantSnapshotsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates an instant snapshot in the specified zone.

      Args:
        request: (ComputeInstantSnapshotsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instantSnapshots.insert', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instantSnapshots', request_field='instantSnapshot', request_type_name='ComputeInstantSnapshotsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of InstantSnapshot resources contained within the specified zone.

      Args:
        request: (ComputeInstantSnapshotsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstantSnapshotList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instantSnapshots.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/instantSnapshots', request_field='', request_type_name='ComputeInstantSnapshotsListRequest', response_type_name='InstantSnapshotList', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeInstantSnapshotsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instantSnapshots.setIamPolicy', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/instantSnapshots/{resource}/setIamPolicy', request_field='zoneSetPolicyRequest', request_type_name='ComputeInstantSnapshotsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on a instantSnapshot in the given zone. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeInstantSnapshotsSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instantSnapshots.setLabels', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instantSnapshots/{resource}/setLabels', request_field='zoneSetLabelsRequest', request_type_name='ComputeInstantSnapshotsSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeInstantSnapshotsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instantSnapshots.testIamPermissions', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/instantSnapshots/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeInstantSnapshotsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)