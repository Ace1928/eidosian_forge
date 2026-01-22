from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
class UpdateSpeaker(neutronv20.UpdateCommand):
    """Update BGP Speaker's information."""
    resource = 'bgp_speaker'

    def add_known_arguments(self, parser):
        parser.add_argument('--name', help=_('Name of the BGP speaker to update.'))
        add_common_arguments(parser)

    def args2body(self, parsed_args):
        body = {}
        args2body_common_arguments(body, parsed_args)
        return {self.resource: body}