from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def _ConstructMasterAuthorizedNetworksTypePatch(enabled, networks, release_track):
    """Constructs an environment patch for Master authorized networks feature.

  Args:
    enabled: bool, whether master authorized networks should be enabled.
    networks: Iterable(string), master authorized networks.
    release_track: base.ReleaseTrack, the release track of command. It dictates
      which Composer client library is used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    config = messages.EnvironmentConfig()
    networks = [] if networks is None else networks
    config.masterAuthorizedNetworksConfig = messages.MasterAuthorizedNetworksConfig(enabled=enabled, cidrBlocks=[messages.CidrBlock(cidrBlock=network) for network in networks])
    return ('config.master_authorized_networks_config', messages.Environment(config=config))