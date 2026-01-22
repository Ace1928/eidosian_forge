import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util
def ValidateMutexOnConfigIdAndParent(args, parent):
    """Validates that only a full resource or resouce args are provided."""
    notification_config_id = args.NOTIFICATIONCONFIGID
    if '/' in notification_config_id:
        if parent is not None:
            raise errors.InvalidNotificationConfigError('Only provide a full resource name (organizations/123/notificationConfigs/test-config) or an --(organization|folder|project) flag, not both.')
    elif parent is None:
        raise errors.InvalidNotificationConfigError('A corresponding parent by a --(organization|folder|project) flag must be provided if it is not included in notification ID.')