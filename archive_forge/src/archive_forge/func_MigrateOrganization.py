from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
def MigrateOrganization(self, request, global_params=None):
    """Migrates a SAS organization to the cloud. This will create GCP projects for each deployment and associate them. The SAS Organization is linked to the gcp project that called the command. go/sas-legacy-customer-migration.

      Args:
        request: (SasPortalMigrateOrganizationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalOperation) The response message.
      """
    config = self.GetMethodConfig('MigrateOrganization')
    return self._RunMethod(config, request, global_params=global_params)