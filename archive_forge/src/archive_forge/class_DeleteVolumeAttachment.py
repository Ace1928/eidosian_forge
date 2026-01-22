import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class DeleteVolumeAttachment(command.Command):
    """Delete an attachment for a volume.

    Similarly to the 'volume attachment create' command, this command will only
    delete the volume attachment record in the Volume service. It will not
    invoke the necessary Compute service actions to actually attach the volume
    to the server at the hypervisor level. As a result, it should typically
    only be used for troubleshooting issues with an existing server in
    combination with other tooling. For all other use cases, the 'server volume
    remove' command should be preferred.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('attachment', metavar='<attachment>', help=_('ID of volume attachment to delete'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.27'):
            msg = _("--os-volume-api-version 3.27 or greater is required to support the 'volume attachment delete' command")
            raise exceptions.CommandError(msg)
        volume_client.attachments.delete(parsed_args.attachment)