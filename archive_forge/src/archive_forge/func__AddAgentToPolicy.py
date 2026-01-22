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
def _AddAgentToPolicy(self, policy, tpu_user_agent):
    """Adds the tpuUserAgent to the policy and return it."""
    logging_binding = None
    storage_binding = None
    tpu_member_str = 'serviceAccount:{}'.format(tpu_user_agent)
    for binding in policy.bindings:
        if binding.role == self.logging_role:
            logging_binding = binding
        if binding.role == self.storage_role:
            storage_binding = binding
        if binding.role != self.tpu_service_agent:
            for member in binding.members:
                if member == tpu_member_str:
                    return None
    if logging_binding is None:
        logging_binding = self.messages.Binding(role=self.logging_role)
        policy.bindings.append(logging_binding)
    if storage_binding is None:
        storage_binding = self.messages.Binding(role=self.storage_role)
        policy.bindings.append(storage_binding)
    logging_binding.members.append(tpu_member_str)
    storage_binding.members.append(tpu_member_str)
    return policy