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
def ModifyNotificationChannel(base_channel, channel_type=None, enabled=None, display_name=None, description=None, field_masks=None):
    """Modifies base_channel's properties using the passed arguments."""
    if field_masks is None:
        field_masks = []
    if channel_type is not None:
        field_masks.append('type')
        base_channel.type = channel_type
    if display_name is not None:
        field_masks.append('display_name')
        base_channel.displayName = display_name
    if description is not None:
        field_masks.append('description')
        base_channel.description = description
    if enabled is not None:
        field_masks.append('enabled')
        base_channel.enabled = enabled
    return base_channel