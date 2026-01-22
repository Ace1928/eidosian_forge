from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import subprocess
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.util.ssh import containers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
import six
Runs a SSH command to a Google Compute Engine VM.

  Args:
    command_list: List with the ssh command to run.
    instance: The GCE VM object.
    user: The user to be used for the SSH command.
    args: The args used to call the gcloud instance.
    ssh_helper: ssh_utils.BaseSSHCLIHelper instance initialized
      for the command.
    explicit_output_file: Use this file for piping stdout of the SSH command,
                          instead of using stdout. This is useful for
                          capturing the command and analyzing it.
    explicit_error_file: Use this file for piping stdout of the SSH command,
                         instead of using stdout. This is useful for
                         capturing the command and analyzing it.
    dry_run: Whether or not this is a dry-run (only print, not run).
  Returns:
    The exit code of the SSH command
  Raises:
    ssh.CommandError: there was an error running a SSH command
  