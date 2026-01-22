import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util
def ValidateAndGetNotificationConfigV2Name(args):
    """Returns relative resource name for a v2 notification config.

  Validates on regexes for args containing full names with locations or short
  names with resources.

  Args:
    args: an argparse object that should contain .NOTIFICATIONCONFIGID,
      optionally 1 of .organization, .folder, .project; and optionally .location

  Examples:

  args with NOTIFICATIONCONFIGID="organizations/123/notificationConfigs/config1"
  and location="locations/us" returns
  organizations/123/locations/us/notificationConfigs/config1

  args with
  NOTIFICATIONCONFIGID="folders/123/locations/us/notificationConfigs/config1"
  and returns folders/123/locations/us/notificationConfigs/config1

  args with NOTIFICATIONCONFIGID="config1", projects="projects/123", and
  locations="us" returns projects/123/notificationConfigs/config1
  """
    id_pattern = re.compile('[a-zA-Z0-9-_]{1,128}$')
    nonregionalized_resource_pattern = re.compile('(organizations|projects|folders)/.+/notificationConfigs/[a-zA-Z0-9-_]{1,128}$')
    regionalized_resource_pattern = re.compile('(organizations|projects|folders)/.+/locations/.+/notificationConfigs/[a-zA-Z0-9-_]{1,128}$')
    notification_config_id = args.NOTIFICATIONCONFIGID
    location = util.ValidateAndGetLocation(args, 'v2')
    if id_pattern.match(notification_config_id):
        return f'{util.GetParentFromNamedArguments(args)}/locations/{location}/notificationConfigs/{notification_config_id}'
    if regionalized_resource_pattern.match(notification_config_id):
        return notification_config_id
    if nonregionalized_resource_pattern.match(notification_config_id):
        [parent_segment, id_segment] = notification_config_id.split('/notificationConfigs/')
        return f'{parent_segment}/locations/{location}/notificationConfigs/{id_segment}'
    raise errors.InvalidNotificationConfigError('NotificationConfig must match (organizations|projects|folders)/.+/notificationConfigs/[a-zA-Z0-9-_]{1,128})$, (organizations|projects|folders)/.+/locations/.+/notificationConfigs/[a-zA-Z0-9-_]{1,128})$, or [a-zA-Z0-9-_]{1,128}$.')