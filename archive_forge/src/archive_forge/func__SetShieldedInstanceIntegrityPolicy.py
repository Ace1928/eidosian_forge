from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _SetShieldedInstanceIntegrityPolicy(holder, client, operation_poller, instance_ref, args):
    """Set the Shielded Instance Integrity Policy."""
    shielded_integrity_policy_msg = client.messages.ShieldedInstanceIntegrityPolicy(updateAutoLearnPolicy=True)
    if not args.IsSpecified('shielded_vm_learn_integrity_policy'):
        return None
    request = client.messages.ComputeInstancesSetShieldedInstanceIntegrityPolicyRequest(instance=instance_ref.Name(), project=instance_ref.project, shieldedInstanceIntegrityPolicy=shielded_integrity_policy_msg, zone=instance_ref.zone)
    operation = client.apitools_client.instances.SetShieldedInstanceIntegrityPolicy(request)
    operation_ref = holder.resources.Parse(operation.selfLink, collection='compute.zoneOperations')
    return waiter.WaitFor(operation_poller, operation_ref, 'Setting shieldedInstanceIntegrityPolicy of instance [{0}]'.format(instance_ref.Name()))