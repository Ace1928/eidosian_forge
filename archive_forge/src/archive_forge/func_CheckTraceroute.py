from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import os
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.diagnose import external_helper
from googlecloudsdk.command_lib.compute.diagnose import internal_helpers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def CheckTraceroute(self, instance, user, dry_run, resource_registry):
    """Checks whether the instance has traceroute in PATH.

    Args:
      instance: Compute Engine VM.
      user: The user to use to SSH into the instance.
      dry_run: Whether to only print commands instead of running them.
      resource_registry: gcloud class used for obtaining data from the
        resources.
    Returns:
      True if the instance has traceroute in PATH,
      False otherwise
    Raises:
      ssh.CommandError: there was an error running a SSH command
    """
    instance_string = internal_helpers.GetInstanceNetworkTitleString(instance)
    log.out.write('Checking traceroute for %s: ' % instance_string)
    if dry_run:
        log.out.Print('[DRY-RUN] No command executed.')
    log.out.flush()
    cmd = ['which', 'traceroute']
    try:
        with files.FileWriter(os.devnull) as dev_null:
            return_code = external_helper.RunSSHCommandToInstance(command_list=cmd, instance=instance, user=user, args=self._args, ssh_helper=self._ssh_helper, explicit_output_file=dev_null, dry_run=dry_run)
    except Exception as e:
        log.out.write(six.text_type(e))
        log.out.write('\n')
        log.out.flush()
        raise ssh.CommandError(' '.join(cmd), six.text_type(e))
    if return_code == 0:
        log.out.Print('Traceroute found in PATH')
    else:
        log.out.Print('Traceroute not found in PATH')
    log.out.flush()
    return return_code == 0