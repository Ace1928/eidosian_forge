import argparse
import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetConsistencyGroup(command.Command):
    _description = _('Set consistency group properties')

    def get_parser(self, prog_name):
        parser = super(SetConsistencyGroup, self).get_parser(prog_name)
        parser.add_argument('consistency_group', metavar='<consistency-group>', help=_('Consistency group to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('New consistency group name'))
        parser.add_argument('--description', metavar='<description>', help=_('New consistency group description'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        kwargs = {}
        if parsed_args.name:
            kwargs['name'] = parsed_args.name
        if parsed_args.description:
            kwargs['description'] = parsed_args.description
        if kwargs:
            consistency_group_id = utils.find_resource(volume_client.consistencygroups, parsed_args.consistency_group).id
            volume_client.consistencygroups.update(consistency_group_id, **kwargs)