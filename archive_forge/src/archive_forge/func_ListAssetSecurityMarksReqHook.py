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
def ListAssetSecurityMarksReqHook(ref, args, req):
    """Retrieves records for a specific asset."""
    del ref
    _ValidateMutexOnAssetAndOrganization(args)
    asset_name = _GetAssetNameForParent(args)
    req.parent = GetParentFromResourceName(asset_name)
    req.filter = 'name="' + asset_name + '"'
    return req