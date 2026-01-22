import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateSecurityGroup(neutronV20.UpdateCommand):
    """Update a given security group."""
    resource = 'security_group'

    def add_known_arguments(self, parser):
        parser.add_argument('--name', help=_('Updated name of the security group.'))
        parser.add_argument('--description', help=_('Updated description of the security group.'))

    def args2body(self, parsed_args):
        body = {}
        neutronV20.update_dict(parsed_args, body, ['name', 'description'])
        return {'security_group': body}