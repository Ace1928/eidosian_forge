from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def CreateSecurityHealthAnalyticsCustomModuleReqHook(ref, args, req):
    """Creates a Security Health Analytics custom module."""
    del ref
    req.parent = _ValidateAndGetParent(args)
    if args.enablement_state not in ['enabled', 'disabled']:
        raise InvalidSCCInputError('Invalid custom module enablement state: %s. Enablement state must be enabled or disabled.' % args.enablement_state)
    return req