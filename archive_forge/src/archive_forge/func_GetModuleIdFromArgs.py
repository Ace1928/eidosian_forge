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
def GetModuleIdFromArgs(args) -> str:
    """Returns the module id from args."""
    if not args.module_id_or_name:
        raise errors.InvalidCustomModuleIdError(None)
    match = _CUSTOM_MODULE_ID_REGEX.fullmatch(args.module_id_or_name)
    if match:
        return match[0]
    else:
        raise errors.InvalidCustomModuleIdError(args.module_id_or_name)