from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def ListEffectiveEventThreatDetectionCustomModulesReqHook(ref, args, req):
    """Lists effective Event Threat Detection custom modules."""
    del ref
    req.parent = _ValidateAndGetParent(args)
    return req