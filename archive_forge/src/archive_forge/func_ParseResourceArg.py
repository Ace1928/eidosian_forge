from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import itertools
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.calliope.concepts import util as resource_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import update_resource_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def ParseResourceArg(self, namespace, group_required=True):
    """Parses the resource ref from namespace (no update flags).

    Args:
      namespace: The argparse namespace.
      group_required: bool, whether parent argument group is required

    Returns:
      The parsed resource ref or None if no resource arg was generated for this
      method.
    """
    if not arg_utils.GetFromNamespace(namespace, self._anchor_name) and (not group_required):
        return None
    result = arg_utils.GetFromNamespace(namespace.CONCEPTS, self._anchor_name)
    if result:
        result = result.Parse()
    if isinstance(result, multitype.TypedConceptResult):
        return result.result
    else:
        return result