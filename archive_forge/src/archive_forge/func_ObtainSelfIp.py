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
def ObtainSelfIp(self, instance, user, dry_run, resource_registry):
    """Returns the localhost ip as seen from the VM.

    Args:
      instance: Compute Engine VM.
      user: The user to use to SSH into the instance.
      dry_run: Whether to only print commands instead of running them.
      resource_registry: gcloud class used for obtaining data from the
        resources.
    Returns:
      A string containing the local ip,
      None if the obtaining was unsuccessful
    Raises:
      ssh.CommandError: there was an error running a SSH command
    """
    instance_string = internal_helpers.GetInstanceNetworkTitleString(instance)
    log.out.write('Obtaining self ip from %s: ' % instance_string)
    log.out.flush()
    if dry_run:
        log.out.Print('<SELF-IP>')
    temp = io.BytesIO()
    cmd = ['echo', '$SSH_CLIENT']
    try:
        external_helper.RunSSHCommandToInstance(command_list=cmd, instance=instance, user=user, args=self._args, ssh_helper=self._ssh_helper, explicit_output_file=temp, dry_run=dry_run)
    except Exception as e:
        log.out.write('\n')
        log.out.flush()
        raise ssh.CommandError(' '.join(cmd), six.text_type(e))
    who_am_i_str = temp.getvalue().decode('utf-8')
    result = re.search('(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})', who_am_i_str)
    if result:
        res = result.group(1)
        log.out.Print(res)
        log.out.flush()
        return res
    return None