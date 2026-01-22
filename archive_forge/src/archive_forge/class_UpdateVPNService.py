from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class UpdateVPNService(neutronv20.UpdateCommand):
    """Update a given VPN service."""
    resource = 'vpnservice'
    help_resource = 'VPN service'

    def add_known_arguments(self, parser):
        add_common_args(parser)
        utils.add_boolean_argument(parser, '--admin-state-up', help=_('Update the admin state for the VPN Service.(True means UP)'))

    def args2body(self, parsed_args):
        body = {}
        common_args2body(parsed_args, body)
        neutronv20.update_dict(parsed_args, body, ['admin_state_up'])
        return {self.resource: body}