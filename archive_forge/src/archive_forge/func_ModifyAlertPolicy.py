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
def ModifyAlertPolicy(base_policy, messages, display_name=None, combiner=None, documentation_content=None, documentation_format=None, enabled=None, channels=None, field_masks=None):
    """Override and/or add fields from other flags to an Alert Policy."""
    if field_masks is None:
        field_masks = []
    if display_name is not None:
        field_masks.append('display_name')
        base_policy.displayName = display_name
    if (documentation_content is not None or documentation_format is not None) and (not base_policy.documentation):
        base_policy.documentation = messages.Documentation()
    if documentation_content is not None:
        field_masks.append('documentation.content')
        base_policy.documentation.content = documentation_content
    if documentation_format is not None:
        field_masks.append('documentation.mime_type')
        base_policy.documentation.mimeType = documentation_format
    if enabled is not None:
        field_masks.append('enabled')
        base_policy.enabled = enabled
    if channels is not None:
        field_masks.append('notification_channels')
        base_policy.notificationChannels = channels
    if combiner is not None:
        field_masks.append('combiner')
        combiner = arg_utils.ChoiceToEnum(combiner, base_policy.CombinerValueValuesEnum, item_type='combiner')
        base_policy.combiner = combiner