from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import util
from googlecloudsdk.core import exceptions as core_exceptions
def _GetNotificationConfigName(args):
    """Returns relative resource name for a notification config."""
    resource_pattern = re.compile('(organizations|projects|folders)/.+/notificationConfigs/[a-zA-Z0-9-_]{1,128}$')
    id_pattern = re.compile('[a-zA-Z0-9-_]{1,128}$')
    if not resource_pattern.match(args.notificationConfigId) and (not id_pattern.match(args.notificationConfigId)):
        raise InvalidNotificationConfigError('NotificationConfig must match either (organizations|projects|folders)/.+/notificationConfigs/[a-zA-Z0-9-_]{1,128})$ or [a-zA-Z0-9-_]{1,128}$.')
    if resource_pattern.match(args.notificationConfigId):
        return args.notificationConfigId
    return util.GetParentFromNamedArguments(args) + '/notificationConfigs/' + args.notificationConfigId