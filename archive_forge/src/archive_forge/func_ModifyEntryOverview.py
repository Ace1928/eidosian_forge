from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
def ModifyEntryOverview(self, request, global_params=None):
    """Modifies entry overview, part of the business context of an Entry. To call this method, you must have the `datacatalog.entries.updateOverview` IAM permission on the corresponding project.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesModifyEntryOverviewRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1EntryOverview) The response message.
      """
    config = self.GetMethodConfig('ModifyEntryOverview')
    return self._RunMethod(config, request, global_params=global_params)