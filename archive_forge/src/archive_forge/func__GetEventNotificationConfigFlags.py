from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from six.moves import map  # pylint: disable=redefined-builtin
from the device. This flag can be specified multiple times to add multiple
def _GetEventNotificationConfigFlags():
    """Returns a list of flags for specfiying Event Notification Configs."""
    event_notification_spec = {'topic': str, 'subfolder': str}
    event_config = base.Argument('--event-notification-config', dest='event_notification_configs', action='append', required=False, type=arg_parsers.ArgDict(spec=event_notification_spec, required_keys=['topic']), help='The configuration for notification of telemetry events received\nfrom the device. This flag can be specified multiple times to add multiple\nconfigs to the device registry. Configs are added to the registry in the order\nthe flags are specified. Only one config with an empty subfolder field is\nallowed and must be specified last.\n\n*topic*:::: A Google Cloud Pub/Sub topic name for event notifications\n\n*subfolder*:::: If the subfolder name matches this string exactly, this\nconfiguration will be used to publish telemetry events. If empty all strings\nare matched.')
    return [event_config]