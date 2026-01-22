import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class CreateMeter(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create network meter')

    def get_parser(self, prog_name):
        parser = super(CreateMeter, self).get_parser(prog_name)
        parser.add_argument('--description', metavar='<description>', help=_('Create description for meter'))
        parser.add_argument('--project', metavar='<project>', help=_("Owner's project (name or ID)"))
        identity_common.add_project_domain_option_to_parser(parser)
        share_group = parser.add_mutually_exclusive_group()
        share_group.add_argument('--share', action='store_true', default=None, help=_('Share meter between projects'))
        share_group.add_argument('--no-share', action='store_true', help=_('Do not share meter between projects'))
        parser.add_argument('name', metavar='<name>', help=_('Name of meter'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        obj = client.create_metering_label(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)