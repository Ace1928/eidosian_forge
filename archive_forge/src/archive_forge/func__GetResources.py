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
def _GetResources(params):
    """Retrieves all resource args from the arg_info tree.

  Args:
    params: an ArgGroup or list of args to parse through.

  Returns:
    YAMLConceptArgument (resource arg) list.
  """
    if isinstance(params, yaml_arg_schema.YAMLConceptArgument):
        return [params]
    if isinstance(params, yaml_arg_schema.Argument):
        return []
    if isinstance(params, yaml_arg_schema.ArgumentGroup):
        params = params.arguments
    result = []
    for param in params:
        result.extend(_GetResources(param))
    return result