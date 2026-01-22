from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudiot import devices
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.command_lib.iot import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import times
import six
def ParseEnableHttpConfig(enable_http_config, client=None):
    if enable_http_config is None:
        return None
    client = client or registries.RegistriesClient()
    http_config_enum = client.http_config_enum
    if enable_http_config:
        return http_config_enum.HTTP_ENABLED
    else:
        return http_config_enum.HTTP_DISABLED