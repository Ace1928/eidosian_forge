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
def _AddUpdatePolicyMostDisruptiveActionFlag(group):
    group.add_argument('--update-policy-most-disruptive-action', choices=InstanceActionChoicesWithNone(flag_prefix='update-policy-'), help='Use this flag to prevent an update if it requires more disruption than you can afford. At most, the MIG performs the specified action on each VM while updating. If the update requires a more disruptive action than the one specified here, then the update fails and no changes are made.')