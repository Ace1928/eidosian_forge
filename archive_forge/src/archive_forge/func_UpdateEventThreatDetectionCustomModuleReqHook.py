from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def UpdateEventThreatDetectionCustomModuleReqHook(ref, args, req):
    """Updates an Event Threat Detection custom module."""
    del ref
    parent = _ValidateAndGetParent(args)
    if parent is None:
        custom_module = _ValidateAndGetCustomModuleFullResourceName(args)
        req.name = custom_module
    else:
        custom_module_id = _ValidateAndGetCustomModuleId(args)
        req.name = parent + '/customModules/' + custom_module_id
    req.updateMask = CleanUpUserInput(req.updateMask)
    if args.enablement_state not in ['enabled', 'disabled', 'inherited']:
        raise InvalidSCCInputError('Invalid custom module enablement state: %s. Enablement state must be enabled, disabled or inherited.' % args.enablement_state)
    return req