import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class DisassociateFlavor(neutronV20.NeutronCommand):
    """Disassociate a Neutron service flavor from a flavor profile."""
    resource = 'flavor'

    def get_parser(self, prog_name):
        parser = super(DisassociateFlavor, self).get_parser(prog_name)
        parser.add_argument('flavor', metavar='FLAVOR', help=_('ID or name of the flavor to be disassociated.'))
        parser.add_argument('flavor_profile', metavar='FLAVOR_PROFILE', help=_('ID of the flavor profile to be disassociated from the flavor.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        flavor_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'flavor', parsed_args.flavor)
        service_profile_id = neutronV20.find_resourceid_by_id(neutron_client, 'service_profile', parsed_args.flavor_profile)
        neutron_client.disassociate_flavor(flavor_id, service_profile_id)
        print(_('Disassociated flavor %(flavor)s from flavor_profile %(profile)s') % {'flavor': parsed_args.flavor, 'profile': parsed_args.flavor_profile}, file=self.app.stdout)