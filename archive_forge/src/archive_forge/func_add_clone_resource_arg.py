from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.transfer.appliances import regions
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_clone_resource_arg(parser):
    """Add a resource argument for cloning a transfer appliance.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser (arg_parse.Parser): The parser for the command.
  """
    concept_parsers.ConceptParser.ForResource('--clone', get_order_resource_spec(), 'The order to clone.', prefixes=True, required=False).AddToParser(parser)