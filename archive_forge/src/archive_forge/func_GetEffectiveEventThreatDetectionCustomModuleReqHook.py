from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def GetEffectiveEventThreatDetectionCustomModuleReqHook(ref, args, req):
    """Gets an effective Event Threat Detection custom module."""
    del ref
    parent = _ValidateAndGetParent(args)
    if parent is None:
        custom_module = _ValidateAndGetEffectiveCustomModuleFullResourceName(args)
        req.name = custom_module
    else:
        custom_module_id = _ValidateAndGetCustomModuleId(args)
        req.name = parent + '/effectiveCustomModules/' + custom_module_id
    return req