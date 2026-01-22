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
def ParseMembership(args, prompt=False, autoselect=False, search=False, flag_override=''):
    """Returns a membership on which to run the command, given the arguments.

  Allows for a `--membership` flag or a `MEMBERSHIP_NAME` positional flag.

  Args:
    args: object containing arguments passed as flags with the command
    prompt: whether to prompt in console for a membership when none are provided
      in args
    autoselect: if no membership is provided and only one exists,
      automatically use that one
    search: whether to search for the membership and error if it does not exist
      (not recommended)
    flag_override: to use a custom membership flag name

  Returns:
    membership: A membership resource name string

  Raises:
    exceptions.Error: no memberships were found or memberships are invalid
    calliope_exceptions.RequiredArgumentException: membership was not provided
  """
    if args.IsKnownAndSpecified('membership') or args.IsKnownAndSpecified('MEMBERSHIP_NAME') or args.IsKnownAndSpecified(flag_override):
        if resources.MembershipLocationSpecified(args, flag_override) or not search:
            return resources.MembershipResourceName(args, flag_override)
        else:
            return resources.SearchMembershipResource(args, flag_override, filter_cluster_missing=True)
    if not prompt and (not autoselect):
        raise MembershipRequiredError(args, flag_override)
    all_memberships, unreachable = api_util.ListMembershipsFull(filter_cluster_missing=True)
    if unreachable:
        raise exceptions.Error('Locations {} are currently unreachable. Please specify memberships using `--location` or the full resource name (projects/*/locations/*/memberships/*)'.format(unreachable))
    if autoselect and len(all_memberships) == 1:
        log.status.Print('Selecting membership [{}].'.format(all_memberships[0]))
        return all_memberships[0]
    if prompt:
        membership = resources.PromptForMembership(all_memberships)
        if membership is not None:
            return membership
    raise MembershipRequiredError(args, flag_override)