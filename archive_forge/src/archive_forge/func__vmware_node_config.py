from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _vmware_node_config(self, args: parser_extensions.Namespace):
    """Constructs proto message VmwareNodeConfig."""
    kwargs = {'cpus': flags.Get(args, 'cpus'), 'memoryMb': flags.Get(args, 'memory'), 'replicas': flags.Get(args, 'replicas'), 'imageType': flags.Get(args, 'image_type'), 'image': flags.Get(args, 'image'), 'bootDiskSizeGb': flags.Get(args, 'boot_disk_size'), 'taints': self._node_taints(args), 'labels': self._labels_value(args), 'vsphereConfig': self._vsphere_config(args), 'enableLoadBalancer': self._enable_load_balancer(args)}
    if flags.IsSet(kwargs):
        return messages.VmwareNodeConfig(**kwargs)
    return None