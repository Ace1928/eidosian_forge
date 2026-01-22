from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import flags as shared_flags
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.ai.persistent_resources import persistent_resource_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddPersistentResourceResourceArg(parser, verb, regions=constants.SUPPORTED_TRAINING_REGIONS):
    """Add a resource argument for a Vertex AI Persistent Resource.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    regions: list[str], the list of supported regions.
  """
    resource_spec = concepts.ResourceSpec(resource_collection=persistent_resource_util.PERSISTENT_RESOURCE_COLLECTION, resource_name='persistent resource', locationsId=shared_flags.RegionAttributeConfig(prompt_func=region_util.GetPromptForRegionFunc(regions)), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)
    concept_parsers.ConceptParser.ForResource('persistent_resource', resource_spec, 'The persistent resource {}.'.format(verb), required=True).AddToParser(parser)