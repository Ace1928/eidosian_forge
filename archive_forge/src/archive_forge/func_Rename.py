from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
def Rename(self, request, global_params=None):
    """Renames a field in a tag template. You must enable the Data Catalog API in the project identified by the `name` parameter. For more information, see [Data Catalog resource project] (https://cloud.google.com/data-catalog/docs/concepts/resource-project).

      Args:
        request: (DatacatalogProjectsLocationsTagTemplatesFieldsRenameRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1TagTemplateField) The response message.
      """
    config = self.GetMethodConfig('Rename')
    return self._RunMethod(config, request, global_params=global_params)