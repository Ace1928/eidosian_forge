from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class InterconnectsService(base_api.BaseApiService):
    """Service class for the interconnects resource."""
    _NAME = 'interconnects'

    def __init__(self, client):
        super(ComputeBeta.InterconnectsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified Interconnect.

      Args:
        request: (ComputeInterconnectsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.interconnects.delete', ordered_params=['project', 'interconnect'], path_params=['interconnect', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/interconnects/{interconnect}', request_field='', request_type_name='ComputeInterconnectsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified Interconnect. Get a list of available Interconnects by making a list() request.

      Args:
        request: (ComputeInterconnectsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Interconnect) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.interconnects.get', ordered_params=['project', 'interconnect'], path_params=['interconnect', 'project'], query_params=[], relative_path='projects/{project}/global/interconnects/{interconnect}', request_field='', request_type_name='ComputeInterconnectsGetRequest', response_type_name='Interconnect', supports_download=False)

    def GetDiagnostics(self, request, global_params=None):
        """Returns the interconnectDiagnostics for the specified Interconnect. In the event of a global outage, do not use this API to make decisions about where to redirect your network traffic. Unlike a VLAN attachment, which is regional, a Cloud Interconnect connection is a global resource. A global outage can prevent this API from functioning properly.

      Args:
        request: (ComputeInterconnectsGetDiagnosticsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InterconnectsGetDiagnosticsResponse) The response message.
      """
        config = self.GetMethodConfig('GetDiagnostics')
        return self._RunMethod(config, request, global_params=global_params)
    GetDiagnostics.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.interconnects.getDiagnostics', ordered_params=['project', 'interconnect'], path_params=['interconnect', 'project'], query_params=[], relative_path='projects/{project}/global/interconnects/{interconnect}/getDiagnostics', request_field='', request_type_name='ComputeInterconnectsGetDiagnosticsRequest', response_type_name='InterconnectsGetDiagnosticsResponse', supports_download=False)

    def GetMacsecConfig(self, request, global_params=None):
        """Returns the interconnectMacsecConfig for the specified Interconnect.

      Args:
        request: (ComputeInterconnectsGetMacsecConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InterconnectsGetMacsecConfigResponse) The response message.
      """
        config = self.GetMethodConfig('GetMacsecConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetMacsecConfig.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.interconnects.getMacsecConfig', ordered_params=['project', 'interconnect'], path_params=['interconnect', 'project'], query_params=[], relative_path='projects/{project}/global/interconnects/{interconnect}/getMacsecConfig', request_field='', request_type_name='ComputeInterconnectsGetMacsecConfigRequest', response_type_name='InterconnectsGetMacsecConfigResponse', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates an Interconnect in the specified project using the data included in the request.

      Args:
        request: (ComputeInterconnectsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.interconnects.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/interconnects', request_field='interconnect', request_type_name='ComputeInterconnectsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of Interconnects available to the specified project.

      Args:
        request: (ComputeInterconnectsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InterconnectList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.interconnects.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/interconnects', request_field='', request_type_name='ComputeInterconnectsListRequest', response_type_name='InterconnectList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified Interconnect with the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeInterconnectsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.interconnects.patch', ordered_params=['project', 'interconnect'], path_params=['interconnect', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/interconnects/{interconnect}', request_field='interconnectResource', request_type_name='ComputeInterconnectsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on an Interconnect. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeInterconnectsSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.interconnects.setLabels', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/interconnects/{resource}/setLabels', request_field='globalSetLabelsRequest', request_type_name='ComputeInterconnectsSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeInterconnectsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.interconnects.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/interconnects/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeInterconnectsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)