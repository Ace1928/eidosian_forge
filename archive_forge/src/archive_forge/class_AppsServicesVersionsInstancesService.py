from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
class AppsServicesVersionsInstancesService(base_api.BaseApiService):
    """Service class for the apps_services_versions_instances resource."""
    _NAME = 'apps_services_versions_instances'

    def __init__(self, client):
        super(AppengineV1beta.AppsServicesVersionsInstancesService, self).__init__(client)
        self._upload_configs = {}

    def Debug(self, request, global_params=None):
        """Enables debugging on a VM instance. This allows you to use the SSH command to connect to the virtual machine where the instance lives. While in "debug mode", the instance continues to serve live traffic. You should delete the instance when you are done debugging and then allow the system to take over and determine if another instance should be started.Only applicable for instances in App Engine flexible environment.

      Args:
        request: (AppengineAppsServicesVersionsInstancesDebugRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Debug')
        return self._RunMethod(config, request, global_params=global_params)
    Debug.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/services/{servicesId}/versions/{versionsId}/instances/{instancesId}:debug', http_method='POST', method_id='appengine.apps.services.versions.instances.debug', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:debug', request_field='debugInstanceRequest', request_type_name='AppengineAppsServicesVersionsInstancesDebugRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Stops a running instance.The instance might be automatically recreated based on the scaling settings of the version. For more information, see "How Instances are Managed" (standard environment (https://cloud.google.com/appengine/docs/standard/python/how-instances-are-managed) | flexible environment (https://cloud.google.com/appengine/docs/flexible/python/how-instances-are-managed)).To ensure that instances are not re-created and avoid getting billed, you can stop all instances within the target version by changing the serving status of the version to STOPPED with the apps.services.versions.patch (https://cloud.google.com/appengine/docs/admin-api/reference/rest/v1/apps.services.versions/patch) method.

      Args:
        request: (AppengineAppsServicesVersionsInstancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/services/{servicesId}/versions/{versionsId}/instances/{instancesId}', http_method='DELETE', method_id='appengine.apps.services.versions.instances.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsServicesVersionsInstancesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets instance information.

      Args:
        request: (AppengineAppsServicesVersionsInstancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Instance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/services/{servicesId}/versions/{versionsId}/instances/{instancesId}', http_method='GET', method_id='appengine.apps.services.versions.instances.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsServicesVersionsInstancesGetRequest', response_type_name='Instance', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the instances of a version.Tip: To aggregate details about instances over time, see the Stackdriver Monitoring API (https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.timeSeries/list).

      Args:
        request: (AppengineAppsServicesVersionsInstancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInstancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/services/{servicesId}/versions/{versionsId}/instances', http_method='GET', method_id='appengine.apps.services.versions.instances.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/instances', request_field='', request_type_name='AppengineAppsServicesVersionsInstancesListRequest', response_type_name='ListInstancesResponse', supports_download=False)