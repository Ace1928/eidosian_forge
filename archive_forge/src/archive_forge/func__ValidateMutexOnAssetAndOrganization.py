from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
from googlecloudsdk.command_lib.scc.hooks import GetOrganization
from googlecloudsdk.command_lib.scc.hooks import GetParentFromResourceName
from googlecloudsdk.command_lib.scc.util import GetParentFromPositionalArguments
def _ValidateMutexOnAssetAndOrganization(args):
    """Validates that only a full resource name or split arguments are provided."""
    if '/' in args.asset and args.organization is not None:
        raise InvalidSCCInputError('Only provide a full resouce name (organizations/123/assets/456) or an --organization flag, not both.')