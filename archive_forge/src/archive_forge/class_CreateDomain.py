import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CreateDomain(command.ShowOne):
    _description = _('Create new domain')

    def get_parser(self, prog_name):
        parser = super(CreateDomain, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<domain-name>', help=_('New domain name'))
        parser.add_argument('--description', metavar='<description>', help=_('New domain description'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', action='store_true', help=_('Enable domain (default)'))
        enable_group.add_argument('--disable', action='store_true', help=_('Disable domain'))
        parser.add_argument('--or-show', action='store_true', help=_('Return existing domain'))
        common.add_resource_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        enabled = True
        if parsed_args.disable:
            enabled = False
        options = common.get_immutable_options(parsed_args)
        try:
            domain = identity_client.domains.create(name=parsed_args.name, description=parsed_args.description, options=options, enabled=enabled)
        except ks_exc.Conflict:
            if parsed_args.or_show:
                domain = utils.find_resource(identity_client.domains, parsed_args.name)
                LOG.info(_('Returning existing domain %s'), domain.name)
            else:
                raise
        domain._info.pop('links')
        return zip(*sorted(domain._info.items()))