from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def SimulateSecurityHealthAnalyticsCustomModulesReqHook(ref, args, req):
    """Simulates a Security Health Analytics custom module."""
    del ref
    req.parent = _ValidateAndGetParent(args)
    return req