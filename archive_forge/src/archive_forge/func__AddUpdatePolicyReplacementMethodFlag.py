from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddUpdatePolicyReplacementMethodFlag(group):
    group.add_argument('--update-policy-replacement-method', choices={'substitute': 'Delete old VMs and create VMs with new names.', 'recreate': 'Recreate VMs and preserve the VM names. The VM IDs and creation timestamps might change.'}, help='Type of replacement method. Specifies what action will be taken to update VMs.')