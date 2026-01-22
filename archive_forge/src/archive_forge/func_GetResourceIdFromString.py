from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.resource_settings import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetResourceIdFromString(setting):
    """Returns the resource id from the setting path.

  A setting path should start with following syntax:
  [organizations|folders|projects]/{resource_id}/settings/{setting_name}/value

  Args:
    setting: A String that contains the setting path
  """
    return setting.split('/')[1]