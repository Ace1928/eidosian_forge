import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class UpdateIPsecSiteConnection(IPsecSiteConnectionMixin, neutronv20.UpdateCommand):
    """Update a given IPsec site connection."""
    resource = 'ipsec_site_connection'
    help_resource = 'IPsec site connection'

    def add_known_arguments(self, parser):
        utils.add_boolean_argument(parser, '--admin-state-up', help=_('Update the administrative state. (True meaning "Up")'))
        super(UpdateIPsecSiteConnection, self).add_known_arguments(parser, False)

    def args2body(self, parsed_args):
        body = {}
        neutronv20.update_dict(parsed_args, body, ['admin_state_up'])
        return super(UpdateIPsecSiteConnection, self).args2body(parsed_args, body)