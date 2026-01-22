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
def CreateFromImport(self, args: parser_extensions.Namespace, bare_metal_cluster, bare_metal_cluster_ref):
    """Creates an Anthos cluster on bare metal."""
    kwargs = {'parent': bare_metal_cluster_ref.Parent().RelativeName(), 'validateOnly': self.GetFlag(args, 'validate_only'), 'bareMetalCluster': bare_metal_cluster, 'bareMetalClusterId': bare_metal_cluster_ref.Name()}
    req = messages.GkeonpremProjectsLocationsBareMetalClustersCreateRequest(**kwargs)
    return self._service.Create(req)