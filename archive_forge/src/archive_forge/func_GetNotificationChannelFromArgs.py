from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def GetNotificationChannelFromArgs(args, messages):
    """Builds a NotificationChannel message from args."""
    channels_base_flags = ['--display-name', '--channel-content', '--channel-content-from-file']
    ValidateAtleastOneSpecified(args, channels_base_flags)
    channel_string = args.channel_content or args.channel_content_from_file
    if channel_string:
        channel = MessageFromString(channel_string, messages.NotificationChannel, 'NotificationChannel', field_remappings=CHANNELS_FIELD_REMAPPINGS)
        if channel.labels:
            channel.labels.additionalProperties = sorted(channel.labels.additionalProperties, key=lambda prop: prop.key)
    else:
        channel = messages.NotificationChannel()
    enabled = args.enabled if args.IsSpecified('enabled') else None
    return ModifyNotificationChannel(channel, channel_type=args.type, display_name=args.display_name, description=args.description, enabled=enabled)