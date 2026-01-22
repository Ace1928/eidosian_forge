from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddNotificationChannelSettingFlags(parser, update=False):
    """Adds flags for channel settings to the parser."""
    channel_group = parser.add_group(help='Notification channel settings')
    AddDisplayNameFlag(channel_group, 'channel')
    channel_group.add_argument('--description', help='An optional description for the channel.')
    channel_group.add_argument('--type', help='The type of the notification channel. This field matches the value of the NotificationChannelDescriptor type field.')
    enabled_kwargs = {'action': arg_parsers.StoreTrueFalseAction if update else 'store_true'}
    if not update:
        enabled_kwargs['default'] = True
    channel_group.add_argument('--enabled', help='Whether notifications are forwarded to the described channel.', **enabled_kwargs)
    if update:
        AddUpdateLabelsFlags('user-labels', channel_group, group_text='User Labels')
        AddUpdateLabelsFlags('channel-labels', channel_group, validate_values=False, group_text='Configuration Fields: Key-Value pairs that define the channel and its behavior.')
    else:
        AddCreateLabelsFlag(channel_group, 'user-labels', 'channel')
        AddCreateLabelsFlag(channel_group, 'channel-labels', 'channel', validate_values=False, extra_message='These are configuration fields that define the channel and its behavior.')