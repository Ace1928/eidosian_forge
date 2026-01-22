from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddAzureServicesAuthentication(auth_config_group, create=True):
    """Adds --azure-tenant-id and --azure-application-id flags."""
    group = auth_config_group.add_argument_group('Azure services authentication')
    group.add_argument('--azure-tenant-id', required=create, help='ID of the Azure Tenant to manage Azure resources.')
    group.add_argument('--azure-application-id', required=create, help='ID of the Azure Application to manage Azure resources.')
    if not create:
        AddClearClient(group)