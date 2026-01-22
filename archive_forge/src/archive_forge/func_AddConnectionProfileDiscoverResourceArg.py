from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddConnectionProfileDiscoverResourceArg(parser):
    """Add a resource argument for a Datastream connection profile discover command.

  Args:
    parser: the parser for the command.
  """
    connection_profile_parser = parser.add_group(mutex=True, required=True)
    connection_profile_parser.add_argument('--connection-profile-object-file', help='Path to a YAML (or JSON) file containing the configuration\n      for a connection profile object. If you pass - as the value of the\n      flag the file content will be read from stdin.')
    resource_specs = [presentation_specs.ResourcePresentationSpec('--connection-profile-name', GetConnectionProfileResourceSpec(), 'Resource ID of the connection profile.', flag_name_overrides={'location': ''}, group=connection_profile_parser)]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--connection-profile-name.location': ['--location']}).AddToParser(parser)