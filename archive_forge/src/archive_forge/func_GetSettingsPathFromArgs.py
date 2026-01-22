from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.resource_settings import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetSettingsPathFromArgs(args):
    """Returns the settings path from the user-specified arguments.

  A settings path has the following syntax:
  [organizations|folders|projects]/{resource_id}/settings/{setting_name}.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
  """
    resource = GetParentResourceFromArgs(args)
    setting_name = GetSettingNameFromArgs(args)
    return '{}/settings/{}'.format(resource, setting_name)