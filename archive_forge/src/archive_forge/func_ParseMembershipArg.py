from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import util as cmd_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def ParseMembershipArg(args, membership_flag='MEMBERSHIP_NAME'):
    """Returns a membership on which to run the command, given the arguments.

  This function is currently only used by the unregister command. This logic
  should be combined with the feature ParseMembership function in a later CL.
  Allows for `MEMBERSHIP_NAME` positional flag.

  Args:
    args: object containing arguments passed as flags with the command
    membership_flag: the membership flag used to pass in the memberhip resource

  Returns:
    membership: A membership resource name string

  Raises:
    exceptions.Error: no memberships were found or memberships are invalid
    calliope_exceptions.RequiredArgumentException: membership was not provided
  """
    if args.IsKnownAndSpecified(membership_flag):
        if MembershipLocationSpecified(args):
            return MembershipResourceName(args)
        else:
            return SearchMembershipResource(args)
    raise calliope_exceptions.RequiredArgumentException(membership_flag, 'membership is required for this command.')