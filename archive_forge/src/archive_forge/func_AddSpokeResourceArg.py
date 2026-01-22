from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.network_connectivity import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddSpokeResourceArg(parser, verb, vpc_spoke_only_command=False):
    """Add a resource argument for a spoke.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    vpc_spoke_only_command: bool, if the spoke resource arg is for a VPC
      spoke-specific command.
  """
    location_arguments = GetResourceLocationArguments(vpc_spoke_only_command)
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='spoke', concept_spec=GetSpokeResourceSpec(location_arguments), required=True, flag_name_overrides={'location': ''}, group_help='Name of the spoke {}.'.format(verb))
    concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)