from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
class FoldersLocationsEventThreatDetectionCustomModulesService(base_api.BaseApiService):
    """Service class for the folders_locations_eventThreatDetectionCustomModules resource."""
    _NAME = 'folders_locations_eventThreatDetectionCustomModules'

    def __init__(self, client):
        super(SecuritycentermanagementV1.FoldersLocationsEventThreatDetectionCustomModulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a resident Event Threat Detection custom module at the scope of the given Resource Manager parent, and also creates inherited custom modules for all descendants of the given parent. These modules are enabled by default.

      Args:
        request: (SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EventThreatDetectionCustomModule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/eventThreatDetectionCustomModules', http_method='POST', method_id='securitycentermanagement.folders.locations.eventThreatDetectionCustomModules.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly'], relative_path='v1/{+parent}/eventThreatDetectionCustomModules', request_field='eventThreatDetectionCustomModule', request_type_name='SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesCreateRequest', response_type_name='EventThreatDetectionCustomModule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified Event Threat Detection custom module and all of its descendants in the Resource Manager hierarchy. This method is only supported for resident custom modules.

      Args:
        request: (SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/eventThreatDetectionCustomModules/{eventThreatDetectionCustomModulesId}', http_method='DELETE', method_id='securitycentermanagement.folders.locations.eventThreatDetectionCustomModules.delete', ordered_params=['name'], path_params=['name'], query_params=['validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an Event Threat Detection custom module.

      Args:
        request: (SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EventThreatDetectionCustomModule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/eventThreatDetectionCustomModules/{eventThreatDetectionCustomModulesId}', http_method='GET', method_id='securitycentermanagement.folders.locations.eventThreatDetectionCustomModules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesGetRequest', response_type_name='EventThreatDetectionCustomModule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all Event Threat Detection custom modules for the given Resource Manager parent. This includes resident modules defined at the scope of the parent along with modules inherited from ancestors.

      Args:
        request: (SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEventThreatDetectionCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/eventThreatDetectionCustomModules', http_method='GET', method_id='securitycentermanagement.folders.locations.eventThreatDetectionCustomModules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/eventThreatDetectionCustomModules', request_field='', request_type_name='SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesListRequest', response_type_name='ListEventThreatDetectionCustomModulesResponse', supports_download=False)

    def ListDescendant(self, request, global_params=None):
        """Lists all resident Event Threat Detection custom modules under the given Resource Manager parent and its descendants.

      Args:
        request: (SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesListDescendantRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDescendantEventThreatDetectionCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('ListDescendant')
        return self._RunMethod(config, request, global_params=global_params)
    ListDescendant.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/eventThreatDetectionCustomModules:listDescendant', http_method='GET', method_id='securitycentermanagement.folders.locations.eventThreatDetectionCustomModules.listDescendant', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/eventThreatDetectionCustomModules:listDescendant', request_field='', request_type_name='SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesListDescendantRequest', response_type_name='ListDescendantEventThreatDetectionCustomModulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the Event Threat Detection custom module with the given name based on the given update mask. Updating the enablement state is supported for both resident and inherited modules (though resident modules cannot have an enablement state of "inherited"). Updating the display name or configuration of a module is supported for resident modules only. The type of a module cannot be changed.

      Args:
        request: (SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EventThreatDetectionCustomModule) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/eventThreatDetectionCustomModules/{eventThreatDetectionCustomModulesId}', http_method='PATCH', method_id='securitycentermanagement.folders.locations.eventThreatDetectionCustomModules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='eventThreatDetectionCustomModule', request_type_name='SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesPatchRequest', response_type_name='EventThreatDetectionCustomModule', supports_download=False)

    def Validate(self, request, global_params=None):
        """Validates the given Event Threat Detection custom module.

      Args:
        request: (SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesValidateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateEventThreatDetectionCustomModuleResponse) The response message.
      """
        config = self.GetMethodConfig('Validate')
        return self._RunMethod(config, request, global_params=global_params)
    Validate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/eventThreatDetectionCustomModules:validate', http_method='POST', method_id='securitycentermanagement.folders.locations.eventThreatDetectionCustomModules.validate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/eventThreatDetectionCustomModules:validate', request_field='validateEventThreatDetectionCustomModuleRequest', request_type_name='SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesValidateRequest', response_type_name='ValidateEventThreatDetectionCustomModuleResponse', supports_download=False)