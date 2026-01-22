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
def _AddUpdatePolicyMaxSurgeFlag(group):
    group.add_argument('--update-policy-max-surge', metavar='MAX_SURGE', type=str, help='Maximum additional number of VMs that can be created during the update process. This can be a fixed number (e.g. 5) or a percentage of size to the managed instance group (e.g. 10%).')