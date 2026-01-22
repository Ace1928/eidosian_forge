import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils as oscutils
class UnsetShareSnapshot(command.Command):
    """Unset a share snapshot property."""
    _description = _('Unset a share snapshot property')

    def get_parser(self, prog_name):
        parser = super(UnsetShareSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of the snapshot to set a property for'))
        parser.add_argument('--name', action='store_true', help=_('Unset snapshot name.'))
        parser.add_argument('--description', action='store_true', help=_('Unset snapshot description.'))
        parser.add_argument('--property', metavar='<key>', action='append', help=_('Remove a property from snapshot (repeat option to remove multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_snapshot = utils.find_resource(share_client.share_snapshots, parsed_args.snapshot)
        kwargs = {}
        if parsed_args.name:
            kwargs['display_name'] = None
        if parsed_args.description:
            kwargs['display_description'] = None
        if kwargs:
            try:
                share_client.share_snapshots.update(share_snapshot, **kwargs)
            except Exception as e:
                raise exceptions.CommandError(_('Failed to unset snapshot display name or display description : %s' % e))
        if parsed_args.property:
            for key in parsed_args.property:
                try:
                    share_snapshot.delete_metadata([key])
                except Exception as e:
                    raise exceptions.CommandError(_("Failed to unset snapshot property '%(key)s': %(e)s"), {'key': key, 'e': e})