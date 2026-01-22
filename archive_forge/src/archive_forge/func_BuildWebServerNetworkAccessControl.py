from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_ALPHA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_BETA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_GA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_ALPHA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_BETA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_GA
def BuildWebServerNetworkAccessControl(web_server_access_control, release_track):
    """Builds a WebServerNetworkAccessControl proto given an IP range list.

  If the list is empty, the returned policy is set to ALLOW by default.
  Otherwise, the default policy is DENY with a list of ALLOW rules for each
  of the IP ranges.

  Args:
    web_server_access_control: [{string: string}], list of IP ranges with
      descriptions.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.

  Returns:
    WebServerNetworkAccessControl: proto to be sent to the API.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    return messages.WebServerNetworkAccessControl(allowedIpRanges=[messages.AllowedIpRange(value=ip_range['ip_range'], description=ip_range.get('description')) for ip_range in web_server_access_control])