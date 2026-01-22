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
def _BuildEnv(entries):
    software_config = messages.SoftwareConfig(envVariables=env_variables_cls(additionalProperties=entries))
    config = messages.EnvironmentConfig(softwareConfig=software_config)
    return env_cls(config=config)