from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import util
from googlecloudsdk.core import exceptions as core_exceptions
def _ValidateMutexOnConfigIdAndParent(args, parent):
    """Validates that only a full resource name or split arguments are provided.
  """
    if '/' in args.notificationConfigId:
        if parent is not None:
            raise InvalidNotificationConfigError('Only provide a full resource name (organizations/123/notificationConfigs/test-config) or an --(organization|folder|project) flag, not both.')
    elif parent is None:
        raise InvalidNotificationConfigError('A corresponding parent by a --(organization|folder|project) flag must be provided if it is not included in notification ID.')