from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def ValidateEventThreatDetectionCustomModulesReqHook(ref, args, req):
    """Validates an Event Threat Detection custom module."""
    del ref
    parent = _ValidateAndGetParent(args)
    custom_module_name = ''
    if parent is None:
        custom_module_name = _ValidateAndGetCustomModuleFullResourceName(args)
    else:
        custom_module_id = _ValidateAndGetCustomModuleId(args)
        custom_module_name = parent + '/customModules/' + custom_module_id
    test_req = req.validateEventThreatDetectionCustomModuleRequest
    req.name = custom_module_name
    if test_req.eventThreatDetectionCustomModule is not None:
        test_req.eventThreatDetectionCustomModule.name = custom_module_name
    return req