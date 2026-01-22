from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def GetEffectiveSecurityHealthAnalyticsCustomModuleReqHook(ref, args, req):
    """Gets an effective Security Health Analytics custom module."""
    del ref
    parent = _ValidateAndGetParent(args)
    if parent is not None:
        custom_module_id = _ValidateAndGetCustomModuleId(args)
        req.name = parent + '/effectiveCustomModules/' + custom_module_id
    else:
        custom_module = _ValidateAndGetEffectiveCustomModuleFullResourceName(args)
        req.name = custom_module
    return req