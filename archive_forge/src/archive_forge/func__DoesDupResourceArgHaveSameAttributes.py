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
def _DoesDupResourceArgHaveSameAttributes(resource, resource_params):
    """Verify if there is a duplicated resource argument with the same attributes.

  Args:
    resource: yaml_arg_schema.Argument, resource to be verified.
    resource_params: [yaml_arg_schema.Argument], list to check duplicate.

  Returns:
    True if there is a duplicate resource arg in the list with same attributes.
  """
    for res_arg in resource_params:
        if res_arg != resource and res_arg.name == resource.name:
            return res_arg.attribute_names == resource.attribute_names
    return False