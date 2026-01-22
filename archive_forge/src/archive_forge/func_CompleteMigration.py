from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.metastore.v1beta import metastore_v1beta_messages as messages
def CompleteMigration(self, request, global_params=None):
    """Completes the managed migration process. The Dataproc Metastore service will switch to using its own backend database after successful migration.

      Args:
        request: (MetastoreProjectsLocationsServicesCompleteMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('CompleteMigration')
    return self._RunMethod(config, request, global_params=global_params)