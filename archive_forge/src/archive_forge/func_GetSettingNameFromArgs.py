from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.resource_settings import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetSettingNameFromArgs(args):
    """Returns the setting name from the user-specified arguments.

  This handles both cases in which the user specifies and does not specify the
  setting prefix.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
  """
    if args.setting_name.startswith(SETTINGS_PREFIX):
        return args.setting_name[len(SETTINGS_PREFIX):]
    return args.setting_name