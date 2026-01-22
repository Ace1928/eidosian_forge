from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddPrivateConnectionResourceArg(parser, verb, release_track, positional=True):
    """Add a resource argument for a Datastream private connection.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    release_track: Some arguments are added based on the command release
      track.
    positional: bool, if True, means that the resource is a positional rather
      than a flag.
  """
    if positional:
        name = 'private_connection'
    else:
        name = '--private-connection'
    vpc_peering_config_parser = parser.add_group(required=True)
    vpc_peering_config_parser.add_argument('--subnet', help='A free subnet for peering. (CIDR of /29).', required=True)
    vpc_field_name = 'vpc'
    if release_track == base.ReleaseTrack.BETA:
        vpc_field_name = 'vpc-name'
    resource_specs = [presentation_specs.ResourcePresentationSpec(name, GetPrivateConnectionResourceSpec(), 'The private connection {}.'.format(verb), required=True), presentation_specs.ResourcePresentationSpec('--%s' % vpc_field_name, GetVpcResourceSpec(), 'Resource ID of the private connection.', group=vpc_peering_config_parser, required=True)]
    concept_parsers.ConceptParser(resource_specs).AddToParser(parser)