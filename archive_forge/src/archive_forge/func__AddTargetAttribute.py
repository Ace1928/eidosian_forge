from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _AddTargetAttribute(messages, resource_selector, message):
    """Add a new TargetAttribute to the resource selector resource."""
    target_attribute = messages.TargetAttribute()
    for field in message:
        value = message.get(field)
        if field == ID_FIELD:
            setattr(target_attribute, field, value)
        if field == LABELS_FIELD:
            deploy_util.SetMetadata(messages, target_attribute, deploy_util.ResourceType.TARGET_ATTRIBUTE, None, value)
        resource_selector.targets.append(target_attribute)