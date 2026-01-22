from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator, Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.api_lib.container.vmware import version_util
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _vmware_vcenter_config(self, args: parser_extensions.Namespace):
    """Constructs proto message VmwareVCenterConfig."""
    kwargs = {'caCertData': flags.Get(args, 'vcenter_ca_cert_data'), 'cluster': flags.Get(args, 'vcenter_cluster'), 'datacenter': flags.Get(args, 'vcenter_datacenter'), 'datastore': flags.Get(args, 'vcenter_datastore'), 'folder': flags.Get(args, 'vcenter_folder'), 'resourcePool': flags.Get(args, 'vcenter_resource_pool'), 'storagePolicyName': flags.Get(args, 'vcenter_storage_policy_name')}
    if flags.IsSet(kwargs):
        return messages.VmwareVCenterConfig(**kwargs)
    return None