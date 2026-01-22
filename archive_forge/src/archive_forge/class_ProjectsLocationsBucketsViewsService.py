from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class ProjectsLocationsBucketsViewsService(base_api.BaseApiService):
    """Service class for the projects_locations_buckets_views resource."""
    _NAME = 'projects_locations_buckets_views'

    def __init__(self, client):
        super(LoggingV2.ProjectsLocationsBucketsViewsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a view over log entries in a log bucket. A bucket may contain a maximum of 30 views.

      Args:
        request: (LoggingProjectsLocationsBucketsViewsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogView) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/buckets/{bucketsId}/views', http_method='POST', method_id='logging.projects.locations.buckets.views.create', ordered_params=['parent'], path_params=['parent'], query_params=['viewId'], relative_path='v2/{+parent}/views', request_field='logView', request_type_name='LoggingProjectsLocationsBucketsViewsCreateRequest', response_type_name='LogView', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a view on a log bucket. If an UNAVAILABLE error is returned, this indicates that system is not in a state where it can delete the view. If this occurs, please try again in a few minutes.

      Args:
        request: (LoggingProjectsLocationsBucketsViewsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/buckets/{bucketsId}/views/{viewsId}', http_method='DELETE', method_id='logging.projects.locations.buckets.views.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingProjectsLocationsBucketsViewsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a view on a log bucket.

      Args:
        request: (LoggingProjectsLocationsBucketsViewsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogView) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/buckets/{bucketsId}/views/{viewsId}', http_method='GET', method_id='logging.projects.locations.buckets.views.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingProjectsLocationsBucketsViewsGetRequest', response_type_name='LogView', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (LoggingProjectsLocationsBucketsViewsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/buckets/{bucketsId}/views/{viewsId}:getIamPolicy', http_method='POST', method_id='logging.projects.locations.buckets.views.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='LoggingProjectsLocationsBucketsViewsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists views on a log bucket.

      Args:
        request: (LoggingProjectsLocationsBucketsViewsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListViewsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/buckets/{bucketsId}/views', http_method='GET', method_id='logging.projects.locations.buckets.views.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/views', request_field='', request_type_name='LoggingProjectsLocationsBucketsViewsListRequest', response_type_name='ListViewsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a view on a log bucket. This method replaces the value of the filter field from the existing view with the corresponding value from the new view. If an UNAVAILABLE error is returned, this indicates that system is not in a state where it can update the view. If this occurs, please try again in a few minutes.

      Args:
        request: (LoggingProjectsLocationsBucketsViewsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogView) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/buckets/{bucketsId}/views/{viewsId}', http_method='PATCH', method_id='logging.projects.locations.buckets.views.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='logView', request_type_name='LoggingProjectsLocationsBucketsViewsPatchRequest', response_type_name='LogView', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.Can return NOT_FOUND, INVALID_ARGUMENT, and PERMISSION_DENIED errors.

      Args:
        request: (LoggingProjectsLocationsBucketsViewsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/buckets/{bucketsId}/views/{viewsId}:setIamPolicy', http_method='POST', method_id='logging.projects.locations.buckets.views.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='LoggingProjectsLocationsBucketsViewsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (LoggingProjectsLocationsBucketsViewsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/buckets/{bucketsId}/views/{viewsId}:testIamPermissions', http_method='POST', method_id='logging.projects.locations.buckets.views.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='LoggingProjectsLocationsBucketsViewsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)