from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import text
def ClusterUpgradeMessage(name, server_conf=None, cluster=None, master=False, node_pool_name=None, new_version=None, new_image_type=None, new_machine_type=None, new_disk_type=None, new_disk_size=None):
    """Get a message to print during gcloud container clusters upgrade.

  Args:
    name: str, the name of the cluster being upgraded.
    server_conf: the server config object.
    cluster: the cluster object.
    master: bool, if the upgrade applies to the master version.
    node_pool_name: str, the name of the node pool if the upgrade is for a
      specific node pool.
    new_version: str, the name of the new version, if given.
    new_image_type: str, the name of the new node image type, if given.
    new_machine_type: str, the name of the new machine type, if given.
    new_disk_type: str, the name of the new boot disk type, if given.
    new_disk_size: int, the size of the new boot disk in GB, if given.

  Raises:
    NodePoolError: if the node pool name can't be found in the cluster.

  Returns:
    str, a message about which nodes in the cluster will be upgraded and
        to which version.
  """
    if master:
        upgrade_message = _MasterUpgradeMessage(name, server_conf, cluster, new_version)
    else:
        upgrade_message = _NodeUpgradeMessage(name, cluster, node_pool_name, new_version, new_image_type, new_machine_type, new_disk_type, new_disk_size)
    return '{} This operation is long-running and will block other operations on the cluster (including delete) until it has run to completion.'.format(upgrade_message)