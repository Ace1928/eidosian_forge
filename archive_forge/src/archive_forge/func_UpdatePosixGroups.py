from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.organizations import org_utils
import six
def UpdatePosixGroups(unused_ref, args, request):
    """Update posix groups.

  When adding posix groups, the posix groups in the request will be combined
  with the current posix groups. When removing groups, the current list of
  posix groups is retrieved and if any value in args.remove_posix_groups
  matches either a name or gid in a current posix group, it will be removed
  from the list and the remaining posix groups will be added to the update
  request.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.
  """
    version = GetApiVersion(args)
    group = ci_client.GetGroup(version, request.name)
    if args.IsSpecified('add_posix_group'):
        request.group.posixGroups = request.group.posixGroups + group.posixGroups
    elif args.IsSpecified('remove_posix_groups'):
        if request.group is None:
            request.group = group
        for pg in list(group.posixGroups):
            if six.text_type(pg.gid) in args.remove_posix_groups or pg.name in args.remove_posix_groups:
                group.posixGroups.remove(pg)
        request.group.posixGroups = group.posixGroups
    return request