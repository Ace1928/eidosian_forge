from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import text
def _NodeUpgradeMessage(name, cluster, node_pool_name, new_version, new_image_type, new_machine_type, new_disk_type, new_disk_size):
    """Returns the prompt message during a node upgrade.

  Args:
    name: str, the name of the cluster being upgraded.
    cluster: the cluster object.
    node_pool_name: str, the name of the node pool if the upgrade is for a
      specific node pool.
    new_version: str, the name of the new version, if given.
    new_image_type: str, the name of the new image type, if given.
    new_machine_type: str, the name of the new machine type, if given.
    new_disk_type: str, the name of the new disk type, if given.
    new_disk_size: int, the size of the new disk, if given.

  Raises:
    NodePoolError: if the node pool name can't be found in the cluster.

  Returns:
    str, a message about which nodes in the cluster will be upgraded and
        to which version, image, or config, if applicable.
  """
    node_message = 'All nodes'
    current_version = None
    if node_pool_name:
        node_message = '{} in node pool [{}]'.format(node_message, node_pool_name)
        if cluster:
            current_version = _NodePoolFromCluster(cluster, node_pool_name).version
    elif cluster:
        node_message = '{} ({} {})'.format(node_message, cluster.currentNodeCount, text.Pluralize(cluster.currentNodeCount, 'node'))
        current_version = cluster.currentNodeVersion
    if current_version:
        version_message = 'version [{}]'.format(current_version)
    else:
        version_message = 'its current version'
    if not new_version and cluster:
        new_version = cluster.currentMasterVersion
    if new_version:
        new_version_message = 'version [{}]'.format(new_version)
    else:
        new_version_message = 'the master version'

    def _UpgradeMessage(field, current, new):
        from_current = 'from {}'.format(current) if current else ''
        return '{} of cluster [{}] {} will change {} to {}.'.format(node_message, name, field, from_current, new)
    if new_image_type:
        image_type = None
        if cluster and node_pool_name:
            image_type = _NodePoolFromCluster(cluster, node_pool_name).config.imageType
        if image_type:
            return '{} of cluster [{}] image will change from {} to {}.'.format(node_message, name, image_type, new_image_type)
        else:
            return '{} of cluster [{}] image will change to {}.'.format(node_message, name, new_image_type)
    node_upgrade_messages = []
    if new_machine_type:
        machine_type = None
        if cluster and node_pool_name:
            machine_type = _NodePoolFromCluster(cluster, node_pool_name).config.machineType
        node_upgrade_messages.append(_UpgradeMessage('machine_type', machine_type, new_machine_type))
    if new_disk_type:
        disk_type = None
        if cluster and node_pool_name:
            disk_type = _NodePoolFromCluster(cluster, node_pool_name).config.diskType
        node_upgrade_messages.append(_UpgradeMessage('disk_type', disk_type, new_disk_type))
    if new_disk_size:
        disk_size = None
        if cluster and node_pool_name:
            disk_size = _NodePoolFromCluster(cluster, node_pool_name).config.diskSizeGb
        node_upgrade_messages.append(_UpgradeMessage('disk_size', disk_size, new_disk_size))
    if not node_upgrade_messages:
        return '{} of cluster [{}] will be upgraded from {} to {}.'.format(node_message, name, version_message, new_version_message)
    return ''.join(node_upgrade_messages)