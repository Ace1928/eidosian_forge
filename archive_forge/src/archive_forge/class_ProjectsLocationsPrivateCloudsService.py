from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateCloudsService(base_api.BaseApiService):
    """Service class for the projects_locations_privateClouds resource."""
    _NAME = 'projects_locations_privateClouds'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateCloudsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `PrivateCloud` resource in a given project and location. Private clouds of type `STANDARD` and `TIME_LIMITED` are zonal resources, `STRETCHED` private clouds are regional. Creating a private cloud also creates a [management cluster](https://cloud.google.com/vmware-engine/docs/concepts-vmware-components) for that private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.create', ordered_params=['parent'], path_params=['parent'], query_params=['privateCloudId', 'requestId', 'validateOnly'], relative_path='v1/{+parent}/privateClouds', request_field='privateCloud', request_type_name='VmwareengineProjectsLocationsPrivateCloudsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Schedules a `PrivateCloud` resource for deletion. A `PrivateCloud` resource scheduled for deletion has `PrivateCloud.state` set to `DELETED` and `expireTime` set to the time when deletion is final and can no longer be reversed. The delete operation is marked as done as soon as the `PrivateCloud` is successfully scheduled for deletion (this also applies when `delayHours` is set to zero), and the operation is not kept in pending state until `PrivateCloud` is purged. `PrivateCloud` can be restored using `UndeletePrivateCloud` method before the `expireTime` elapses. When `expireTime` is reached, deletion is final and all private cloud resources are irreversibly removed and billing stops. During the final removal process, `PrivateCloud.state` is set to `PURGING`. `PrivateCloud` can be polled using standard `GET` method for the whole period of deletion and purging. It will not be returned only when it is completely purged.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}', http_method='DELETE', method_id='vmwareengine.projects.locations.privateClouds.delete', ordered_params=['name'], path_params=['name'], query_params=['delayHours', 'force', 'requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a `PrivateCloud` resource by its resource name.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PrivateCloud) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsGetRequest', response_type_name='PrivateCloud', supports_download=False)

    def GetDnsForwarding(self, request, global_params=None):
        """Gets details of the `DnsForwarding` config.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsGetDnsForwardingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsForwarding) The response message.
      """
        config = self.GetMethodConfig('GetDnsForwarding')
        return self._RunMethod(config, request, global_params=global_params)
    GetDnsForwarding.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/dnsForwarding', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.getDnsForwarding', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsGetDnsForwardingRequest', response_type_name='DnsForwarding', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}:getIamPolicy', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `PrivateCloud` resources in a given project and location.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPrivateCloudsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/privateClouds', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsListRequest', response_type_name='ListPrivateCloudsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Modifies a `PrivateCloud` resource. Only the following fields can be updated: `description`. Only fields specified in `updateMask` are applied. During operation processing, the resource is temporarily in the `ACTIVE` state before the operation fully completes. For that period of time, you can't update the resource. Use the operation status to determine when the processing fully completes.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}', http_method='PATCH', method_id='vmwareengine.projects.locations.privateClouds.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='privateCloud', request_type_name='VmwareengineProjectsLocationsPrivateCloudsPatchRequest', response_type_name='Operation', supports_download=False)

    def ResetNsxCredentials(self, request, global_params=None):
        """Resets credentials of the NSX appliance.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsResetNsxCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ResetNsxCredentials')
        return self._RunMethod(config, request, global_params=global_params)
    ResetNsxCredentials.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}:resetNsxCredentials', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.resetNsxCredentials', ordered_params=['privateCloud'], path_params=['privateCloud'], query_params=[], relative_path='v1/{+privateCloud}:resetNsxCredentials', request_field='resetNsxCredentialsRequest', request_type_name='VmwareengineProjectsLocationsPrivateCloudsResetNsxCredentialsRequest', response_type_name='Operation', supports_download=False)

    def ResetVcenterCredentials(self, request, global_params=None):
        """Resets credentials of the Vcenter appliance.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsResetVcenterCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ResetVcenterCredentials')
        return self._RunMethod(config, request, global_params=global_params)
    ResetVcenterCredentials.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}:resetVcenterCredentials', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.resetVcenterCredentials', ordered_params=['privateCloud'], path_params=['privateCloud'], query_params=[], relative_path='v1/{+privateCloud}:resetVcenterCredentials', request_field='resetVcenterCredentialsRequest', request_type_name='VmwareengineProjectsLocationsPrivateCloudsResetVcenterCredentialsRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}:setIamPolicy', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='VmwareengineProjectsLocationsPrivateCloudsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def ShowNsxCredentials(self, request, global_params=None):
        """Gets details of credentials for NSX appliance.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsShowNsxCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Credentials) The response message.
      """
        config = self.GetMethodConfig('ShowNsxCredentials')
        return self._RunMethod(config, request, global_params=global_params)
    ShowNsxCredentials.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}:showNsxCredentials', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.showNsxCredentials', ordered_params=['privateCloud'], path_params=['privateCloud'], query_params=[], relative_path='v1/{+privateCloud}:showNsxCredentials', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsShowNsxCredentialsRequest', response_type_name='Credentials', supports_download=False)

    def ShowVcenterCredentials(self, request, global_params=None):
        """Gets details of credentials for Vcenter appliance.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsShowVcenterCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Credentials) The response message.
      """
        config = self.GetMethodConfig('ShowVcenterCredentials')
        return self._RunMethod(config, request, global_params=global_params)
    ShowVcenterCredentials.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}:showVcenterCredentials', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.showVcenterCredentials', ordered_params=['privateCloud'], path_params=['privateCloud'], query_params=['username'], relative_path='v1/{+privateCloud}:showVcenterCredentials', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsShowVcenterCredentialsRequest', response_type_name='Credentials', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}:testIamPermissions', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='VmwareengineProjectsLocationsPrivateCloudsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Restores a private cloud that was previously scheduled for deletion by `DeletePrivateCloud`. A `PrivateCloud` resource scheduled for deletion has `PrivateCloud.state` set to `DELETED` and `PrivateCloud.expireTime` set to the time when deletion can no longer be reversed.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}:undelete', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeletePrivateCloudRequest', request_type_name='VmwareengineProjectsLocationsPrivateCloudsUndeleteRequest', response_type_name='Operation', supports_download=False)

    def UpdateDnsForwarding(self, request, global_params=None):
        """Updates the parameters of the `DnsForwarding` config, like associated domains. Only fields specified in `update_mask` are applied.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsUpdateDnsForwardingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateDnsForwarding')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateDnsForwarding.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/dnsForwarding', http_method='PATCH', method_id='vmwareengine.projects.locations.privateClouds.updateDnsForwarding', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='dnsForwarding', request_type_name='VmwareengineProjectsLocationsPrivateCloudsUpdateDnsForwardingRequest', response_type_name='Operation', supports_download=False)