from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import random
import re
import string
import sys
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves import range  # pylint: disable=redefined-builtin
def AddAutoscaledPropertyToMigs(migs, client, resources):
    """Add Autoscaler information if Autoscaler is defined for the MIGs.

  Issue additional queries to detect if any given Instange Group Manager is
  a target of some autoscaler and add this information to in 'autoscaled'
  property.

  Args:
    migs: list of dicts, List of IGM resources converted to dictionaries
    client: a GCE client
    resources: a GCE resource registry

  Returns:
    Pair of:
    - boolean - True iff any autoscaler has an error
    - Copy of migs list with additional property 'autoscaled' set to 'No'/'Yes'/
    'Yes (*)' for each MIG depending on look-up result.
  """
    augmented_migs = []
    had_errors = False
    for mig in AddAutoscalersToMigs(migs_iterator=_ComputeInstanceGroupSize(migs, client, resources), client=client, resources=resources, fail_when_api_not_supported=False):
        status = ResolveAutoscalingStatusForMig(client, mig)
        if status == client.messages.Autoscaler.StatusValueValuesEnum.ERROR:
            had_errors = True
        augmented_migs.append(mig)
    return (had_errors, augmented_migs)