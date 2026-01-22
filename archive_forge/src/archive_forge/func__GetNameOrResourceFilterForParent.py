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
def _GetNameOrResourceFilterForParent(args):
    """Returns a filter with either name or resourceName as filter."""
    request_filter = ''
    if args.asset is not None:
        request_filter = 'name="' + _GetAssetNameForParent(args) + '"'
    else:
        request_filter = 'securityCenterProperties.resourceName="' + args.resource_name + '"'
    return request_filter