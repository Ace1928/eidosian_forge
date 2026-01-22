from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as core_api_exceptions
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.features import info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
def MembershipRequiredError(args, flag_override=''):
    """Parses a list of membership resources from args.

  Assumes a `--memberships` flag or a `MEMBERSHIP_NAME` flag unless overridden.

  Args:
    args: argparse.Namespace arguments provided for the command
    flag_override: set to override the name of the membership flag

  Returns:
    memberships: A list of membership resource name strings

  Raises:
    exceptions.Error: if no memberships were found or memberships are invalid
    calliope_exceptions.RequiredArgumentException: if membership was not
      provided
  """
    if flag_override:
        flag = flag_override
    elif args.IsKnownAndSpecified('MEMBERSHIP_NAME'):
        flag = 'MEMBERSHIP_NAME'
    else:
        flag = 'memberships'
    return calliope_exceptions.RequiredArgumentException(flag, 'At least one membership is required for this command.')