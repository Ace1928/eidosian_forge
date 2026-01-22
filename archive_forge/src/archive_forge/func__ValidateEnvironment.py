from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
@staticmethod
def _ValidateEnvironment(env_obj, release_track):
    messages = api_util.GetMessagesModule(release_track=release_track)
    if env_obj.config.resilienceMode is None or env_obj.config.resilienceMode == messages.EnvironmentConfig.ResilienceModeValueValuesEnum.RESILIENCE_MODE_UNSPECIFIED:
        raise command_util.InvalidUserInputError('Cannot trigger a database failover for environments without enabled high resilience mode.')