import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util
def GetParentFromNotificationConfigName(notification_config_name):
    resource_pattern = re.compile('(organizations|projects|folders)/.*')
    if not resource_pattern.match(notification_config_name):
        raise errors.InvalidSCCInputError('When providing a full resource path, it must also include the pattern the organization, project, or folder prefix.')
    return notification_config_name.split('/notificationConfigs/')[0]