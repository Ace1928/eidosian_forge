from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
class ServicesProjectSettingsService(base_api.BaseApiService):
    """Service class for the services_projectSettings resource."""
    _NAME = u'services_projectSettings'

    def __init__(self, client):
        super(ServicemanagementV1.ServicesProjectSettingsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves the settings that control the specified consumer project's usage.
of the service.

      Args:
        request: (ServicemanagementServicesProjectSettingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectSettings) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'servicemanagement.services.projectSettings.get', ordered_params=[u'serviceName', u'consumerProjectId'], path_params=[u'consumerProjectId', u'serviceName'], query_params=[u'expand', u'view'], relative_path=u'v1/services/{serviceName}/projectSettings/{consumerProjectId}', request_field='', request_type_name=u'ServicemanagementServicesProjectSettingsGetRequest', response_type_name=u'ProjectSettings', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates specified subset of the settings that control the specified.
consumer project's usage of the service.  Attempts to update a field not
controlled by the caller will result in an access denied error.

Operation<response: ProjectSettings>

      Args:
        request: (ServicemanagementServicesProjectSettingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'servicemanagement.services.projectSettings.patch', ordered_params=[u'serviceName', u'consumerProjectId'], path_params=[u'consumerProjectId', u'serviceName'], query_params=[u'updateMask'], relative_path=u'v1/services/{serviceName}/projectSettings/{consumerProjectId}', request_field=u'projectSettings', request_type_name=u'ServicemanagementServicesProjectSettingsPatchRequest', response_type_name=u'Operation', supports_download=False)

    def Update(self, request, global_params=None):
        """NOTE: Currently unsupported.  Use PatchProjectSettings instead.

Updates the settings that control the specified consumer project's usage
of the service.  Attempts to update a field not controlled by the caller
will result in an access denied error.

Operation<response: ProjectSettings>

      Args:
        request: (ProjectSettings) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'servicemanagement.services.projectSettings.update', ordered_params=[u'serviceName', u'consumerProjectId'], path_params=[u'consumerProjectId', u'serviceName'], query_params=[], relative_path=u'v1/services/{serviceName}/projectSettings/{consumerProjectId}', request_field='<request>', request_type_name=u'ProjectSettings', response_type_name=u'Operation', supports_download=False)