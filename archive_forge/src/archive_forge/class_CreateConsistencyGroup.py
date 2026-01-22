import argparse
import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateConsistencyGroup(command.ShowOne):
    _description = _('Create new consistency group.')

    def get_parser(self, prog_name):
        parser = super(CreateConsistencyGroup, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', nargs='?', help=_('Name of new consistency group (default to None)'))
        exclusive_group = parser.add_mutually_exclusive_group(required=True)
        exclusive_group.add_argument('--volume-type', metavar='<volume-type>', help=_('Volume type of this consistency group (name or ID)'))
        exclusive_group.add_argument('--source', metavar='<consistency-group>', help=_('Existing consistency group (name or ID)'))
        exclusive_group.add_argument('--consistency-group-source', metavar='<consistency-group>', dest='source', help=argparse.SUPPRESS)
        exclusive_group.add_argument('--snapshot', metavar='<consistency-group-snapshot>', help=_('Existing consistency group snapshot (name or ID)'))
        exclusive_group.add_argument('--consistency-group-snapshot', metavar='<consistency-group-snapshot>', dest='snapshot', help=argparse.SUPPRESS)
        parser.add_argument('--description', metavar='<description>', help=_('Description of this consistency group'))
        parser.add_argument('--availability-zone', metavar='<availability-zone>', help=_('Availability zone for this consistency group (not available if creating consistency group from source)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if parsed_args.volume_type:
            volume_type_id = utils.find_resource(volume_client.volume_types, parsed_args.volume_type).id
            consistency_group = volume_client.consistencygroups.create(volume_type_id, name=parsed_args.name, description=parsed_args.description, availability_zone=parsed_args.availability_zone)
        else:
            if parsed_args.availability_zone:
                msg = _("'--availability-zone' option will not work if creating consistency group from source")
                LOG.warning(msg)
            consistency_group_id = None
            consistency_group_snapshot = None
            if parsed_args.source:
                consistency_group_id = utils.find_resource(volume_client.consistencygroups, parsed_args.source).id
            elif parsed_args.snapshot:
                consistency_group_snapshot = utils.find_resource(volume_client.cgsnapshots, parsed_args.snapshot).id
            consistency_group = volume_client.consistencygroups.create_from_src(consistency_group_snapshot, consistency_group_id, name=parsed_args.name, description=parsed_args.description)
        return zip(*sorted(consistency_group._info.items()))