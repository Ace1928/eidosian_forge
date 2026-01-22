from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import util
from googlecloudsdk.core import exceptions as core_exceptions
def UpdateNotificationReqHook(ref, args, req):
    """Generate a notification config using organization and config id."""
    del ref
    parent = util.GetParentFromNamedArguments(args)
    _ValidateMutexOnConfigIdAndParent(args, parent)
    req.name = _GetNotificationConfigName(args)
    args.filter = None
    return req