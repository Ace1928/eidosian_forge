from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
class ProjectsLocationsAutoscalingPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_autoscalingPolicies resource."""
    _NAME = 'projects_locations_autoscalingPolicies'

    def __init__(self, client):
        super(DataprocV1.ProjectsLocationsAutoscalingPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates new autoscaling policy.

      Args:
        request: (DataprocProjectsLocationsAutoscalingPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AutoscalingPolicy) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/autoscalingPolicies', http_method='POST', method_id='dataproc.projects.locations.autoscalingPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/autoscalingPolicies', request_field='autoscalingPolicy', request_type_name='DataprocProjectsLocationsAutoscalingPoliciesCreateRequest', response_type_name='AutoscalingPolicy', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an autoscaling policy. It is an error to delete an autoscaling policy that is in use by one or more clusters.

      Args:
        request: (DataprocProjectsLocationsAutoscalingPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/autoscalingPolicies/{autoscalingPoliciesId}', http_method='DELETE', method_id='dataproc.projects.locations.autoscalingPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsLocationsAutoscalingPoliciesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves autoscaling policy.

      Args:
        request: (DataprocProjectsLocationsAutoscalingPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AutoscalingPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/autoscalingPolicies/{autoscalingPoliciesId}', http_method='GET', method_id='dataproc.projects.locations.autoscalingPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsLocationsAutoscalingPoliciesGetRequest', response_type_name='AutoscalingPolicy', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DataprocProjectsLocationsAutoscalingPoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/autoscalingPolicies/{autoscalingPoliciesId}:getIamPolicy', http_method='POST', method_id='dataproc.projects.locations.autoscalingPolicies.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='DataprocProjectsLocationsAutoscalingPoliciesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists autoscaling policies in the project.

      Args:
        request: (DataprocProjectsLocationsAutoscalingPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAutoscalingPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/autoscalingPolicies', http_method='GET', method_id='dataproc.projects.locations.autoscalingPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/autoscalingPolicies', request_field='', request_type_name='DataprocProjectsLocationsAutoscalingPoliciesListRequest', response_type_name='ListAutoscalingPoliciesResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.Can return NOT_FOUND, INVALID_ARGUMENT, and PERMISSION_DENIED errors.

      Args:
        request: (DataprocProjectsLocationsAutoscalingPoliciesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/autoscalingPolicies/{autoscalingPoliciesId}:setIamPolicy', http_method='POST', method_id='dataproc.projects.locations.autoscalingPolicies.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DataprocProjectsLocationsAutoscalingPoliciesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DataprocProjectsLocationsAutoscalingPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/autoscalingPolicies/{autoscalingPoliciesId}:testIamPermissions', http_method='POST', method_id='dataproc.projects.locations.autoscalingPolicies.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DataprocProjectsLocationsAutoscalingPoliciesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates (replaces) autoscaling policy.Disabled check for update_mask, because all updates will be full replacements.

      Args:
        request: (AutoscalingPolicy) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AutoscalingPolicy) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/autoscalingPolicies/{autoscalingPoliciesId}', http_method='PUT', method_id='dataproc.projects.locations.autoscalingPolicies.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='AutoscalingPolicy', response_type_name='AutoscalingPolicy', supports_download=False)