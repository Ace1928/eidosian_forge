from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def UpdateFromFile(self, args: parser_extensions.Namespace, bare_metal_cluster):
    top_level_mutable_fields = ['description', 'bare_metal_version', 'annotations', 'network_config', 'control_plane', 'load_balancer', 'storage', 'proxy', 'cluster_operations', 'maintenance_config', 'node_config', 'security_config', 'node_access_config', 'os_environment_config']
    kwargs = {'name': self._user_cluster_name(args), 'allowMissing': self.GetFlag(args, 'allow_missing'), 'updateMask': ','.join(top_level_mutable_fields), 'validateOnly': self.GetFlag(args, 'validate_only'), 'bareMetalCluster': bare_metal_cluster}
    req = messages.GkeonpremProjectsLocationsBareMetalClustersPatchRequest(**kwargs)
    return self._service.Patch(req)