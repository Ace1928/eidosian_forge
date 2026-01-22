from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
def ModifyColumnFamilies(self, request, global_params=None):
    """Performs a series of column family modifications on the specified table. Either all or none of the modifications will occur before this method returns, but data requests received prior to that point may see a table where only some modifications have taken effect.

      Args:
        request: (BigtableadminProjectsInstancesTablesModifyColumnFamiliesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
    config = self.GetMethodConfig('ModifyColumnFamilies')
    return self._RunMethod(config, request, global_params=global_params)