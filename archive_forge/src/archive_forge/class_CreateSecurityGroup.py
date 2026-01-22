import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateSecurityGroup(neutronV20.CreateCommand):
    """Create a security group."""
    resource = 'security_group'

    def add_known_arguments(self, parser):
        parser.add_argument('name', metavar='NAME', help=_('Name of the security group to be created.'))
        parser.add_argument('--description', help=_('Description of the security group to be created.'))

    def args2body(self, parsed_args):
        body = {'name': parsed_args.name}
        neutronV20.update_dict(parsed_args, body, ['description', 'tenant_id'])
        return {'security_group': body}