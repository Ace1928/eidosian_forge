import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
class UnsetShareGroupSnapshot(command.Command):
    """Unset a share group snapshot property."""
    _description = _('Unset a share group snapshot property')

    def get_parser(self, prog_name):
        parser = super(UnsetShareGroupSnapshot, self).get_parser(prog_name)
        parser.add_argument('share_group_snapshot', metavar='<share-group-snapshot>', help=_('Name or ID of the group snapshot to unset a property of'))
        parser.add_argument('--name', action='store_true', help=_('Unset share group snapshot name.'))
        parser.add_argument('--description', action='store_true', help=_('Unset share group snapshot description.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_group_snapshot = osc_utils.find_resource(share_client.share_group_snapshots, parsed_args.share_group_snapshot)
        kwargs = {}
        if parsed_args.name:
            kwargs['name'] = ''
        if parsed_args.description:
            kwargs['description'] = ''
        if kwargs:
            try:
                share_client.share_group_snapshots.update(share_group_snapshot, **kwargs)
            except Exception as e:
                raise exceptions.CommandError(f'Failed to unset name or description for share group snapshot : {e}')