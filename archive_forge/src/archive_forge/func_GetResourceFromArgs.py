from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.command_lib.org_policies import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetResourceFromArgs(args):
    """Returns the resource from the user-specified arguments.

  A resource has the following syntax:
  [organizations|folders|projects]/{resource_id}.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
  """
    resource_id = args.organization or args.folder or args.project
    if args.organization:
        resource_type = 'organizations'
    elif args.folder:
        resource_type = 'folders'
    else:
        resource_type = 'projects'
    return '{}/{}'.format(resource_type, resource_id)