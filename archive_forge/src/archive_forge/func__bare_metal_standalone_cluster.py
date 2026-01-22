from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _bare_metal_standalone_cluster(self, args: parser_extensions.Namespace):
    """Constructs proto message Bare Metal Standalone Cluster."""
    kwargs = {'name': self._standalone_cluster_name(args), 'description': getattr(args, 'description', None), 'bareMetalVersion': getattr(args, 'version', None), 'networkConfig': self._network_config(args), 'loadBalancer': self._load_balancer_config(args), 'controlPlane': self._control_plane_config(args), 'clusterOperations': self._cluster_operations_config(args), 'maintenanceConfig': self._maintenance_config(args), 'securityConfig': self._security_config(args), 'nodeAccessConfig': self._node_access_config(args), 'annotations': self._annotations(args), 'binaryAuthorization': self._binary_authorization(args)}
    return self._set_config_if_exists(messages.BareMetalStandaloneCluster, kwargs)