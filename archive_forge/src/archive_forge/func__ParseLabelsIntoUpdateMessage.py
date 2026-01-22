from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import update
from googlecloudsdk.command_lib.util.apis import yaml_arg_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_property
def _ParseLabelsIntoUpdateMessage(message, args, api_field):
    """Find diff between existing labels and args, set labels into the message."""
    diff = labels_util.Diff.FromUpdateArgs(args)
    if not diff.MayHaveUpdates():
        return False
    existing_labels = _RetrieveFieldValueFromMessage(message, api_field)
    label_cls = _GetLabelsClass(message, api_field)
    update_result = diff.Apply(label_cls, existing_labels)
    if update_result.needs_update:
        arg_utils.SetFieldInMessage(message, api_field, update_result.labels)
    return True