from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.azure import resource_args
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _Client(self, client_ref, args):
    kwargs = {'applicationId': getattr(args, 'app_id', None), 'name': client_ref.azureClientsId, 'tenantId': getattr(args, 'tenant_id', None)}
    return self._messages.GoogleCloudGkemulticloudV1AzureClient(**kwargs) if any(kwargs.values()) else None