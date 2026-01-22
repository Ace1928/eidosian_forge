from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.resource_settings import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetParentResourceFromString(setting):
    """Returns the resource from the user-specified arguments.

  A setting path should start with following syntax:
  [organizations|folders|projects]/{resource_id}/settings/{setting_name}/value

  Args:
    setting: A String that contains the setting path
  """
    resource_type = setting.split('/')[0]
    resource_id = setting.split('/')[1]
    return '{}/{}'.format(resource_type, resource_id)