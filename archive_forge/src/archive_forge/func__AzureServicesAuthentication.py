from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.azure import resource_args
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _AzureServicesAuthentication(self, args):
    kwargs = {'applicationId': flags.GetAzureApplicationID(args), 'tenantId': flags.GetAzureTenantID(args)}
    if not any(kwargs.values()):
        return None
    return self._messages.GoogleCloudGkemulticloudV1AzureServicesAuthentication(**kwargs)