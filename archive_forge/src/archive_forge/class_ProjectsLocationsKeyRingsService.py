from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.kms_apitools.cloudkms_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
class ProjectsLocationsKeyRingsService(base_api.BaseApiService):
    """Service class for the projects_locations_keyRings resource."""
    _NAME = u'projects_locations_keyRings'

    def __init__(self, client):
        super(CloudkmsV1.ProjectsLocationsKeyRingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new KeyRing in a given Project and Location.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (KeyRing) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.create', ordered_params=[u'parent'], path_params=[u'parent'], query_params=[u'keyRingId'], relative_path=u'v1/{+parent}/keyRings', request_field=u'keyRing', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCreateRequest', response_type_name=u'KeyRing', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns metadata for a given KeyRing.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (KeyRing) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}', http_method=u'GET', method_id=u'cloudkms.projects.locations.keyRings.get', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}', request_field='', request_type_name=u'CloudkmsProjectsLocationsKeyRingsGetRequest', response_type_name=u'KeyRing', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource.
Returns an empty policy if the resource exists and does not have a policy
set.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}:getIamPolicy', http_method=u'GET', method_id=u'cloudkms.projects.locations.keyRings.getIamPolicy', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:getIamPolicy', request_field='', request_type_name=u'CloudkmsProjectsLocationsKeyRingsGetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists KeyRings.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListKeyRingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings', http_method=u'GET', method_id=u'cloudkms.projects.locations.keyRings.list', ordered_params=[u'parent'], path_params=[u'parent'], query_params=[u'pageSize', u'pageToken'], relative_path=u'v1/{+parent}/keyRings', request_field='', request_type_name=u'CloudkmsProjectsLocationsKeyRingsListRequest', response_type_name=u'ListKeyRingsResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any.
existing policy.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}:setIamPolicy', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.setIamPolicy', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:setIamPolicy', request_field=u'setIamPolicyRequest', request_type_name=u'CloudkmsProjectsLocationsKeyRingsSetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.
If the resource does not exist, this will return an empty set of
permissions, not a NOT_FOUND error.

Note: This operation is designed to be used for building permission-aware
UIs and command-line tools, not for authorization checking. This operation
may "fail open" without warning.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}:testIamPermissions', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.testIamPermissions', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:testIamPermissions', request_field=u'testIamPermissionsRequest', request_type_name=u'CloudkmsProjectsLocationsKeyRingsTestIamPermissionsRequest', response_type_name=u'TestIamPermissionsResponse', supports_download=False)