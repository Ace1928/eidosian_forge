from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def ConvertEmailToMembershipResourceName(version, args, group_arg_name, member_arg_name):
    """Convert email to membership resource name.

  Args:
    version: Release track information
    args: The argparse namespace
    group_arg_name: argument/parameter name related to group info
    member_arg_name: argument/parameter name related to member info

  Returns:
    Membership Id (e.g. groups/11zu0gzc3tkdgn2/memberships/1044279104595057141)

  """
    group_id = groups_hooks.ConvertEmailToResourceName(version, args.group_email, group_arg_name)
    try:
        return ci_client.LookupMembershipName(version, group_id, args.member_email).name
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError):
        parameter_name = group_arg_name + ', ' + member_arg_name
        error_msg = 'There is no such membership associated with the specified arguments: {}, {}'.format(args.group_email, args.member_email)
        raise exceptions.InvalidArgumentException(parameter_name, error_msg)