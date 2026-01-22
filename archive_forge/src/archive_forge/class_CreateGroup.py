import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CreateGroup(command.ShowOne):
    _description = _('Create new group')

    def get_parser(self, prog_name):
        parser = super(CreateGroup, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<group-name>', help=_('New group name'))
        parser.add_argument('--domain', metavar='<domain>', help=_('Domain to contain new group (name or ID)'))
        parser.add_argument('--description', metavar='<description>', help=_('New group description'))
        parser.add_argument('--or-show', action='store_true', help=_('Return existing group'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        domain = None
        if parsed_args.domain:
            domain = common.find_domain(identity_client, parsed_args.domain).id
        try:
            group = identity_client.groups.create(name=parsed_args.name, domain=domain, description=parsed_args.description)
        except ks_exc.Conflict:
            if parsed_args.or_show:
                group = utils.find_resource(identity_client.groups, parsed_args.name, domain_id=domain)
                LOG.info(_('Returning existing group %s'), group.name)
            else:
                raise
        group._info.pop('links')
        return zip(*sorted(group._info.items()))