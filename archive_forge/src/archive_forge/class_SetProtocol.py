import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetProtocol(command.Command):
    _description = _('Set federation protocol properties')

    def get_parser(self, prog_name):
        parser = super(SetProtocol, self).get_parser(prog_name)
        parser.add_argument('federation_protocol', metavar='<name>', help=_('Federation protocol to modify (name or ID)'))
        parser.add_argument('--identity-provider', metavar='<identity-provider>', required=True, help=_('Identity provider that supports <federation-protocol> (name or ID) (required)'))
        parser.add_argument('--mapping', metavar='<mapping>', help=_('Mapping that is to be used (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        protocol = identity_client.federation.protocols.update(parsed_args.identity_provider, parsed_args.federation_protocol, parsed_args.mapping)
        info = dict(protocol._info)
        info['identity_provider'] = parsed_args.identity_provider
        info['mapping'] = info.pop('mapping_id')
        return zip(*sorted(info.items()))