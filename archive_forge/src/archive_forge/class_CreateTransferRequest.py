import logging
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateTransferRequest(command.ShowOne):
    _description = _('Create volume transfer request.')

    def get_parser(self, prog_name):
        parser = super(CreateTransferRequest, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help=_('New transfer request name (default to None)'))
        parser.add_argument('--snapshots', action='store_true', dest='snapshots', help=_('Allow transfer volumes without snapshots (default) (supported by --os-volume-api-version 3.55 or later)'), default=None)
        parser.add_argument('--no-snapshots', action='store_false', dest='snapshots', help=_('Disallow transfer volumes without snapshots (supported by --os-volume-api-version 3.55 or later)'))
        parser.add_argument('volume', metavar='<volume>', help=_('Volume to transfer (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        kwargs = {}
        if parsed_args.snapshots is not None:
            if volume_client.api_version < api_versions.APIVersion('3.55'):
                msg = _("--os-volume-api-version 3.55 or greater is required to support the '--(no-)snapshots' option")
                raise exceptions.CommandError(msg)
            kwargs['no_snapshots'] = not parsed_args.snapshots
        volume_id = utils.find_resource(volume_client.volumes, parsed_args.volume).id
        volume_transfer_request = volume_client.transfers.create(volume_id, parsed_args.name, **kwargs)
        volume_transfer_request._info.pop('links', None)
        return zip(*sorted(volume_transfer_request._info.items()))