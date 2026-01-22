from __future__ import absolute_import
from apitools.base.py import base_api
from samples.fusiontables_sample.fusiontables_v1 import fusiontables_v1_messages as messages
class TemplateService(base_api.BaseApiService):
    """Service class for the template resource."""
    _NAME = u'template'

    def __init__(self, client):
        super(FusiontablesV1.TemplateService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a template.

      Args:
        request: (FusiontablesTemplateDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FusiontablesTemplateDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'fusiontables.template.delete', ordered_params=[u'tableId', u'templateId'], path_params=[u'tableId', u'templateId'], query_params=[], relative_path=u'tables/{tableId}/templates/{templateId}', request_field='', request_type_name=u'FusiontablesTemplateDeleteRequest', response_type_name=u'FusiontablesTemplateDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a specific template by its id.

      Args:
        request: (FusiontablesTemplateGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Template) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'fusiontables.template.get', ordered_params=[u'tableId', u'templateId'], path_params=[u'tableId', u'templateId'], query_params=[], relative_path=u'tables/{tableId}/templates/{templateId}', request_field='', request_type_name=u'FusiontablesTemplateGetRequest', response_type_name=u'Template', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new template for the table.

      Args:
        request: (Template) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Template) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'fusiontables.template.insert', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[], relative_path=u'tables/{tableId}/templates', request_field='<request>', request_type_name=u'Template', response_type_name=u'Template', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of templates.

      Args:
        request: (FusiontablesTemplateListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TemplateList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'fusiontables.template.list', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[u'maxResults', u'pageToken'], relative_path=u'tables/{tableId}/templates', request_field='', request_type_name=u'FusiontablesTemplateListRequest', response_type_name=u'TemplateList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing template. This method supports patch semantics.

      Args:
        request: (Template) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Template) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'fusiontables.template.patch', ordered_params=[u'tableId', u'templateId'], path_params=[u'tableId', u'templateId'], query_params=[], relative_path=u'tables/{tableId}/templates/{templateId}', request_field='<request>', request_type_name=u'Template', response_type_name=u'Template', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing template.

      Args:
        request: (Template) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Template) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'fusiontables.template.update', ordered_params=[u'tableId', u'templateId'], path_params=[u'tableId', u'templateId'], query_params=[], relative_path=u'tables/{tableId}/templates/{templateId}', request_field='<request>', request_type_name=u'Template', response_type_name=u'Template', supports_download=False)