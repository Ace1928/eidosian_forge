from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.filestore.backups import util as backup_util
from googlecloudsdk.command_lib.filestore.snapshots import util as snapshot_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseConnectMode(self, network_config, key):
    """Parse and match the supplied connection mode."""
    try:
        value = self.messages.NetworkConfig.ConnectModeValueValuesEnum.lookup_by_name(key)
    except KeyError:
        raise InvalidArgumentError('[{}] is not a valid connect-mode. Must be one of DIRECT_PEERING or PRIVATE_SERVICE_ACCESS.'.format(key))
    else:
        network_config.connectMode = value