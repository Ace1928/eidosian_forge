from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import service as settings_service
def ComputeResourceType(args):
    """Returns the resource type from the user-specified arguments.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
  """
    if args.organization:
        resource_type = ORGANIZATION
    elif args.folder:
        resource_type = FOLDER
    else:
        resource_type = PROJECT
    return resource_type