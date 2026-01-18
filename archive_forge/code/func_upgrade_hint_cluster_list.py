from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import locations as api_util
from googlecloudsdk.command_lib.container.gkemulticloud import constants
def upgrade_hint_cluster_list(platform):
    """Generates a message that warns users if their cluster version can be upgraded.

  Args:
    platform: A string, the platform the component is on {AWS,Azure}.

  Returns:
    A message in how to upgrade a cluster if its end of life.
  """
    upgrade_message = _UPGRADE_CLUSTER_HINT
    upgrade_message += _UPGRADE_COMMAND_CLUSTER.format(USERS_PLATFORM=platform.lower(), LOCATION='LOCATION', CLUSTER_VERSION='CLUSTER_VERSION', CLUSTER_NAME='CLUSTER_NAME')
    upgrade_message += _SUPPORTED_COMMAND.format(USERS_PLATFORM=platform.lower(), LOCATION='LOCATION')
    return upgrade_message