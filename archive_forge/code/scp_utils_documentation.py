from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.util.ssh import ip
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
SCP files between local and remote GCE instance.

    Run this method from subclasses' Run methods.

    Args:
      compute_holder: The ComputeApiHolder.
      args: argparse.Namespace, the args the command was invoked with.
      on_prem: bool, Whether to connect to an on-prem IP.
      port: str or None, Port number to use for SSH connection.
      recursive: bool, Whether to use recursive copying using -R flag.
      compress: bool, Whether to use compression.
      extra_flags: [str] or None, extra flags to add to command invocation.
      release_track: obj, The current release track.
      ip_type: IpTypeEnum, Specify using internal ip or external ip address.

    Raises:
      ssh_utils.NetworkError: Network issue which likely is due to failure
        of SSH key propagation.
      ssh.CommandError: The SSH command exited with SSH exit code, which
        usually implies that a connection problem occurred.
    