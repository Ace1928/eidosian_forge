from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.core import log
Construct an SshTunnelArgs from command line args and values.

  Args:
    args: The parsed commandline arguments. May or may not have had
      AddSshTunnelArgs called.
    api_client: An appengine_api_client.AppEngineApiClient.
    track: ReleaseTrack, The currently running release track.
    project: str, the project id (string with dashes).
    version: The target version reference object.
    instance: The target instance reference object.

  Returns:
    SshTunnelArgs or None if IAP Tunnel is disabled.
  