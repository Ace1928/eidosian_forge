from neutronclient._i18n import _
from neutronclient.common import extension
class FoxInSocketsUpdate(extension.ClientExtensionUpdate, FoxInSocket):
    """Update a fox socket."""
    shell_command = 'fox-sockets-update'
    list_columns = ['id', 'name']

    def add_known_arguments(self, parser):
        parser.add_argument('--name', help=_('Name of this fox socket.'))

    def args2body(self, parsed_args):
        body = {'name': parsed_args.name}
        return {'fox_socket': body}