from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
import sys
import time
from apitools.base.py import list_pager
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import times
import six
def _WaitForSSHKeysToPropagate(self, ssh_helper, remote, identity_file, user, instance, options, putty_force_connect=False):
    """Waits for SSH keys to propagate in order to SSH to the instance."""
    ssh_helper.EnsureSSHKeyExists(self.client, user, instance, ssh_helper.GetProject(self.client, properties.VALUES.core.project.Get(required=True)), times.Now() + datetime.timedelta(seconds=300))
    ssh_poller = ssh.SSHPoller(remote=remote, identity_file=identity_file, options=options, max_wait_ms=300 * 1000)
    try:
        ssh_poller.Poll(ssh_helper.env, putty_force_connect=putty_force_connect)
    except retry.WaitException:
        raise ssh_utils.NetworkError()