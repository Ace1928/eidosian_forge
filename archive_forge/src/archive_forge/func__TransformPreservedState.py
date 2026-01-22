from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def _TransformPreservedState(instance):
    """Transform for the PRESERVED_STATE field in the table output.

  PRESERVED_STATE is generated from the fields preservedStateFromPolicy and
  preservedStateFromConfig fields in the managedInstance message.

  Args:
    instance: instance dictionary for transform

  Returns:
    Preserved state status as one of ('POLICY', 'CONFIG', 'POLICY,CONFIG')
  """
    preserved_state_value = ''
    if 'preservedStateFromPolicy' in instance and instance['preservedStateFromPolicy']:
        preserved_state_value += 'POLICY,'
    if 'preservedStateFromConfig' in instance and instance['preservedStateFromConfig']:
        preserved_state_value += 'CONFIG'
    if preserved_state_value.endswith(','):
        preserved_state_value = preserved_state_value[:-1]
    return preserved_state_value