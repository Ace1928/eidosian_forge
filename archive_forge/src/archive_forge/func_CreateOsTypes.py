from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import sys
from googlecloudsdk.core.resource import resource_property
def CreateOsTypes(os_types):
    """Create Os Types in Ops Agent Policy.

  Args:
    os_types: dict, VM OS type matchers, or None.

  Returns:
    A list of OpsAgentPolicy.Assignment.OsType objects.
  """
    OsType = OpsAgentPolicy.Assignment.OsType
    return [OsType(OsType.OsShortName(os_type['short-name']), os_type['version']) for os_type in os_types or []]