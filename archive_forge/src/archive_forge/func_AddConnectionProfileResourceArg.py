from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddConnectionProfileResourceArg(parser, verb, release_track, positional=True, required=True):
    """Add a resource argument for a Datastream connection profile.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    release_track: Some arguments are added based on the command release
        track.
    positional: bool, if True, means that the resource is a positional rather
      than a flag.
    required: bool, if True, means that a flag is required.
  """
    if positional:
        name = 'connection_profile'
    else:
        name = '--connection-profile'
    connectivity_parser = parser.add_group(mutex=True)
    connectivity_parser.add_argument('--static-ip-connectivity', action='store_true', help='use static ip connectivity')
    if release_track == base.ReleaseTrack.BETA:
        connectivity_parser.add_argument('--no-connectivity', action='store_true', help='no connectivity')
    forward_ssh_parser = connectivity_parser.add_group()
    forward_ssh_parser.add_argument('--forward-ssh-hostname', help='Hostname for the SSH tunnel.', required=required)
    forward_ssh_parser.add_argument('--forward-ssh-username', help='Username for the SSH tunnel.', required=required)
    forward_ssh_parser.add_argument('--forward-ssh-port', help='Port for the SSH tunnel, default value is 22.', type=int, default=22)
    password_group = forward_ssh_parser.add_group(required=required, mutex=True)
    password_group.add_argument('--forward-ssh-password', help='          SSH password.\n          ')
    password_group.add_argument('--forward-ssh-private-key', help='SSH private key..')
    private_connection_flag_name = 'private-connection'
    if release_track == base.ReleaseTrack.BETA:
        private_connection_flag_name = 'private-connection-name'
    resource_specs = [presentation_specs.ResourcePresentationSpec(name, GetConnectionProfileResourceSpec(), 'The connection profile {}.'.format(verb), required=True), presentation_specs.ResourcePresentationSpec('--%s' % private_connection_flag_name, GetPrivateConnectionResourceSpec(), 'Resource ID of the private connection.', flag_name_overrides={'location': ''}, group=connectivity_parser)]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--%s.location' % private_connection_flag_name: ['--location']}).AddToParser(parser)