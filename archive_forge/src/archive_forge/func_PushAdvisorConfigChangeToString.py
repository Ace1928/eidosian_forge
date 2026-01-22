from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def PushAdvisorConfigChangeToString(config_change):
    """Convert a ConfigChange message to a printable string.

  Args:
    config_change: The ConfigChange message to convert.

  Returns:
    An easily readable string representing the ConfigChange message.
  """
    result = 'Element [{element}] (old value = {old_value}, new value = {new_value}) was {change_type}. Advice:\n'.format(element=config_change.element, old_value=config_change.oldValue, new_value=config_change.newValue, change_type=PushAdvisorChangeTypeToString(config_change.changeType))
    for advice in config_change.advices:
        result += '\t* {0}'.format(advice.description)
    return result