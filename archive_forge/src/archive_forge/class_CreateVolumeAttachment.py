import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class CreateVolumeAttachment(command.ShowOne):
    """Create an attachment for a volume.

    This command will only create a volume attachment in the Volume service. It
    will not invoke the necessary Compute service actions to actually attach
    the volume to the server at the hypervisor level. As a result, it should
    typically only be used for troubleshooting issues with an existing server
    in combination with other tooling. For all other use cases, the 'server
    add volume' command should be preferred.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('volume', metavar='<volume>', help=_('Name or ID of volume to attach to server.'))
        parser.add_argument('server', metavar='<server>', help=_('Name or ID of server to attach volume to.'))
        parser.add_argument('--connect', action='store_true', dest='connect', default=False, help=_('Make an active connection using provided connector info'))
        parser.add_argument('--no-connect', action='store_false', dest='connect', help=_('Do not make an active connection using provided connector info'))
        parser.add_argument('--initiator', metavar='<initiator>', help=_('IQN of the initiator attaching to'))
        parser.add_argument('--ip', metavar='<ip>', help=_('IP of the system attaching to'))
        parser.add_argument('--host', metavar='<host>', help=_('Name of the host attaching to'))
        parser.add_argument('--platform', metavar='<platform>', help=_('Platform type'))
        parser.add_argument('--os-type', metavar='<ostype>', help=_('OS type'))
        parser.add_argument('--multipath', action='store_true', dest='multipath', default=False, help=_('Use multipath'))
        parser.add_argument('--no-multipath', action='store_false', dest='multipath', help=_('Use multipath'))
        parser.add_argument('--mountpoint', metavar='<mountpoint>', help=_('Mountpoint volume will be attached at'))
        parser.add_argument('--mode', metavar='<mode>', help=_('Mode of volume attachment, rw, ro and null, where null indicates we want to honor any existing admin-metadata settings (supported by --os-volume-api-version 3.54 or later)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        compute_client = self.app.client_manager.compute
        if volume_client.api_version < api_versions.APIVersion('3.27'):
            msg = _("--os-volume-api-version 3.27 or greater is required to support the 'volume attachment create' command")
            raise exceptions.CommandError(msg)
        if parsed_args.mode:
            if volume_client.api_version < api_versions.APIVersion('3.54'):
                msg = _("--os-volume-api-version 3.54 or greater is required to support the '--mode' option")
                raise exceptions.CommandError(msg)
        connector = {}
        if parsed_args.connect:
            connector = {'initiator': parsed_args.initiator, 'ip': parsed_args.ip, 'platform': parsed_args.platform, 'host': parsed_args.host, 'os_type': parsed_args.os_type, 'multipath': parsed_args.multipath, 'mountpoint': parsed_args.mountpoint}
        elif any({parsed_args.initiator, parsed_args.ip, parsed_args.platform, parsed_args.host, parsed_args.host, parsed_args.multipath, parsed_args.mountpoint}):
            msg = _('You must specify the --connect option for any of the connection-specific options such as --initiator to be valid')
            raise exceptions.CommandError(msg)
        volume = utils.find_resource(volume_client.volumes, parsed_args.volume)
        server = utils.find_resource(compute_client.servers, parsed_args.server)
        attachment = volume_client.attachments.create(volume.id, connector, server.id, parsed_args.mode)
        return _format_attachment(attachment)