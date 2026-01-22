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
def _GetHostKeyFromInstance(self, zone, ssh_helper, instance):
    """Wrapper around SSH Utils to get the host keys for SSH."""
    instance_ref = instance_flags.SSH_INSTANCE_RESOLVER.ResolveResources([instance.name], compute_scope.ScopeEnum.ZONE, zone, self.resources, scope_lister=instance_flags.GetInstanceZoneScopeLister(self.client))[0]
    project = ssh_helper.GetProject(self.client, instance_ref.project)
    host_keys = ssh_helper.GetHostKeysFromGuestAttributes(self.client, instance_ref, instance, project)
    if host_keys is not None and (not host_keys):
        log.status.Print('Unable to retrieve host keys from instance metadata. Continuing.')
    return host_keys