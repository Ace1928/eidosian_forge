from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from openstackclient.i18n import _
class SetAccount(command.Command):
    _description = _('Set account properties')

    def get_parser(self, prog_name):
        parser = super(SetAccount, self).get_parser(prog_name)
        parser.add_argument('--property', metavar='<key=value>', required=True, action=parseractions.KeyValueAction, help=_('Set a property on this account (repeat option to set multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        self.app.client_manager.object_store.account_set(properties=parsed_args.property)