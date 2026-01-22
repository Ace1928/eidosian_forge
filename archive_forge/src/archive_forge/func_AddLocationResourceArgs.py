from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddLocationResourceArgs(parser):
    """Add the location resource argument.

  Args:
    parser: the parser for the command.
  """
    arg_specs = [presentation_specs.ResourcePresentationSpec('--location', GetLocationResourceSpec(), 'The Batch location resource. If you omit this flag, the defaultlocation is used if you set the batch/location property.Otherwise, omitting this flag lists jobs across all locations.', required=False)]
    concept_parsers.ConceptParser(arg_specs).AddToParser(parser)