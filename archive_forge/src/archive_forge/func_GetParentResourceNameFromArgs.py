import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.command_lib.scc.manage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
def GetParentResourceNameFromArgs(args) -> str:
    """Returns the relative path to the parent from args.

  Args:
    args: command line args.

  Returns:
    The relative path. e.g. 'projects/foo/locations/global',
    'folders/1234/locations/global'.
  """
    if args.parent:
        return f'{_ParseParent(args.parent).RelativeName()}/locations/global'
    return f'{_GetParentResourceFromArgs(args).RelativeName()}/locations/global'