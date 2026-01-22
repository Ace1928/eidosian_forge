from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import service as settings_service
def GetRequestFromArgs(args, setting_name, is_effective):
    """Returns the get_request from the user-specified arguments.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
    setting_name: setting name such as `settings/iam-projectCreatorRoles`
    is_effective: indicate if it is requesting for an effective setting
  """
    resource_type = ComputeResourceType(args)
    return GetRequestFromResourceType(resource_type, setting_name, is_effective)