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
def _ConstructHighResiliencePatch(enabled, release_track):
    """Constructs a patch for updating high resilience.

  Args:
    enabled: bool, whether High resilience should be enabled.
    release_track: base.ReleaseTrack, the release track of command. It dictates
      which Composer client library is used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    if not enabled:
        return ('config.resilience_mode', messages.Environment(config=messages.EnvironmentConfig()))
    return ('config.resilience_mode', messages.Environment(config=messages.EnvironmentConfig(resilienceMode=messages.EnvironmentConfig.ResilienceModeValueValuesEnum.HIGH_RESILIENCE)))